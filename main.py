import os
import json
import glob
import pathlib
import logging
import re
import threading
from typing import Dict, List, Optional, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from urllib.parse import quote
from fastapi import Body

# ======================
# 日志：静音 /api/path_exists
# ======================

logger = logging.getLogger("uvicorn.access")


class PathExistsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/api/path_exists" not in msg


logger.addFilter(PathExistsFilter())

# ======================
# CPU 绑核相关（子进程）
# ======================

_CPU_LOCK = threading.Lock()
_CPU_INDEX = 0


def _get_cpu_pool() -> List[int]:
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
    else:
        n = os.cpu_count() or 1
        cpus = list(range(n))
    # 跳过 cpu0（如果还有别的核）
    if 0 in cpus and len(cpus) > 1:
        cpus.remove(0)
    return list(cpus)


CPU_POOL = _get_cpu_pool()


def pick_next_cpu() -> int:
    global _CPU_INDEX
    if not CPU_POOL:
        return 0
    with _CPU_LOCK:
        cpu = CPU_POOL[_CPU_INDEX % len(CPU_POOL)]
        _CPU_INDEX += 1
        return cpu


def make_affinity_preexec(cpu: int):
    # 在子进程 exec 前设置 CPU affinity
    def _fn():
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {cpu})
        except Exception:
            pass

    return _fn


def _get_affinity_of_pid(pid: int):
    try:
        if hasattr(os, "sched_getaffinity"):
            return sorted(os.sched_getaffinity(pid))
    except Exception:
        pass
    # 兜底：读 /proc
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


# ======================
# 路径与全局状态
# ======================

BASE_DIR = pathlib.Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

GLOBAL_STATE_FILE = STORAGE_DIR / "global_state.json"
PROCESS_FILE = BASE_DIR / "process_template.py"

print(
    "[BOOT]",
    "pid=",
    os.getpid(),
    "BASE_DIR=",
    str(BASE_DIR.resolve()),
    "GLOBAL_STATE_FILE=",
    str(GLOBAL_STATE_FILE.resolve()),
    "cwd=",
    os.getcwd(),
)

_STATE_LOCK = threading.Lock()

# ======================
# Pydantic 模型定义
# ======================


class Directory(BaseModel):
    id: str
    name: str
    parent_id: Optional[str] = None
    collapsed: bool = False
    display_index: Optional[int] = None


class Task(BaseModel):
    id: str
    name: str
    directory_id: Optional[str] = None
    # 任务级变量：路径/参数里可用 {{var}}
    vars: Dict[str, Any] = Field(default_factory=dict)
    display_index: Optional[int] = None


class FilePathInfo(BaseModel):
    file_count: int
    total_lines: int


class FilePath(BaseModel):
    id: str
    path: str
    info: Optional[FilePathInfo] = None


class FileGroup(BaseModel):
    id: str
    name: str
    task_id: str
    file_paths: List[FilePath] = Field(default_factory=list)
    collapsed: bool = False
    display_index: Optional[int] = None


class OperationNode(BaseModel):
    id: str
    node_type: str  # 'root' or 'op'
    name: str
    function_name: Optional[str] = None
    params_json: str = "{}"
    description: Optional[str] = None
    result_save_path: Optional[str] = None
    executed: bool = False
    parent_id: Optional[str] = None
    frozen: bool = False


class OperationSequence(BaseModel):
    id: str
    name: str
    task_id: str
    file_group_id: str
    code_save_path: Optional[str] = None
    # 兼容旧字段
    result_save_path: Optional[str] = None
    nodes: List[OperationNode] = Field(default_factory=list)
    show_meta: bool = False
    display_index: Optional[int] = None
    collapsed: bool = False


class AppState(BaseModel):
    version: int = 0
    directories: Dict[str, Directory] = Field(default_factory=dict)
    tasks: Dict[str, Task] = Field(default_factory=dict)
    file_groups: Dict[str, FileGroup] = Field(default_factory=dict)
    operation_sequences: Dict[str, OperationSequence] = Field(default_factory=dict)


class FilePathRequest(BaseModel):
    path: Any  # 允许 string 或 {"$ref":...}；前端一般传 string
    task_id: Optional[str] = None
    sequence_id: Optional[str] = None
    node_id: Optional[str] = None


class SequenceActionRequest(BaseModel):
    task_id: str
    sequence_id: str


class FileStructureResponse(BaseModel):
    success: bool
    structure: Optional[Any] = None
    error: Optional[str] = None


class FileInfoResponse(BaseModel):
    success: bool
    file_count: Optional[int] = None
    total_lines: Optional[int] = None
    error: Optional[str] = None


class HeadTailResponse(BaseModel):
    success: bool
    first_line: Optional[str] = None
    last_line: Optional[str] = None
    error: Optional[str] = None


class PathExistsRequest(BaseModel):
    path: Any  # 允许 string 或 {"$ref":...}
    task_id: Optional[str] = None
    sequence_id: Optional[str] = None
    node_id: Optional[str] = None


class PathExistsResponse(BaseModel):
    success: bool
    exists: bool
    error: Optional[str] = None


# ======================
# 状态读写工具函数
# ======================


def default_state() -> AppState:
    root_dir = Directory(id="root", name="根目录", parent_id=None, collapsed=False)
    return AppState(directories={"root": root_dir})


def _patch_sequences_parent_id(state: AppState) -> None:
    """
    给老版本（没有分支结构的）操作序列补 parent_id：
    - 如果一个 sequence 中，所有 op 节点的 parent_id 都是 None，则认为是旧的线性链：
        第 0 个节点 parent_id = None，之后每个节点的 parent_id = 前一个节点的 id
    - 如果已经有任意一个 op 节点的 parent_id 非 None，则视为新版本数据，不做修改。
    """
    for seq in state.operation_sequences.values():
        nodes = seq.nodes or []
        if not nodes:
            continue

        has_branch = any((n.node_type == "op") and (n.parent_id is not None) for n in nodes)
        if has_branch:
            continue

        for idx, node in enumerate(nodes):
            node.parent_id = None if idx == 0 else nodes[idx - 1].id


def load_state() -> AppState:
    if GLOBAL_STATE_FILE.exists():
        with GLOBAL_STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        state = AppState(**data)
        if "root" not in state.directories:
            state.directories["root"] = Directory(id="root", name="根目录", parent_id=None, collapsed=False)
        _patch_sequences_parent_id(state)
        return state
    return default_state()


def state_to_json(state: AppState) -> str:
    try:
        data = state.model_dump()
    except AttributeError:
        data = state.dict()
    return json.dumps(data, indent=2, ensure_ascii=False)


def model_to_dict(m: BaseModel) -> Dict[str, Any]:
    try:
        return m.model_dump()
    except AttributeError:
        return m.dict()


def sanitize_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch in r'<>:"/\\|?*':
            continue
        if ord(ch) < 32:
            continue
        keep.append(ch)
    cleaned = "".join(keep).strip()
    return cleaned or "task"


def save_state(state: AppState) -> None:
    """
    1) 原子写 global_state.json
    2) 导出每任务 task.json（清理旧的再写新的）
    """
    tmp_path = GLOBAL_STATE_FILE.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(state_to_json(state))
    os.replace(tmp_path, GLOBAL_STATE_FILE)

    for p in STORAGE_DIR.glob("*.task.json"):
        try:
            p.unlink()
        except Exception:
            pass

    for task_id, task in state.tasks.items():
        safe_name = sanitize_filename(task.name or "task")
        file_path = STORAGE_DIR / f"{task_id}_{safe_name}.task.json"

        task_file_groups = [fg for fg in state.file_groups.values() if fg.task_id == task_id]
        task_sequences = [seq for seq in state.operation_sequences.values() if seq.task_id == task_id]

        per_task_data = {
            "task": model_to_dict(task),
            "directories": [model_to_dict(d) for d in state.directories.values()],
            "tasks": [model_to_dict(t) for t in state.tasks.values()],
            "file_groups": [model_to_dict(fg) for fg in task_file_groups],
            "operation_sequences": [model_to_dict(seq) for seq in task_sequences],
        }
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(per_task_data, f, ensure_ascii=False, indent=2)


APP_STATE: AppState = load_state()

# ======================
# 文件结构与信息分析工具
# ======================


def first_non_empty_line_in_file(file_path: str) -> Optional[str]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                return line.rstrip("\r\n")
    return None


def last_non_empty_line_in_file(file_path: str, chunk_size: int = 8192) -> Optional[str]:
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if file_size == 0:
            return None

        buffer = b""
        pos = file_size

        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)

            buffer = chunk + buffer
            lines = buffer.split(b"\n")
            buffer = lines[0]

            for line in reversed(lines[1:]):
                if line.strip():
                    return line.rstrip(b"\r").decode("utf-8", errors="ignore")

        if buffer.strip():
            return buffer.rstrip(b"\r").decode("utf-8", errors="ignore")

    return None


def describe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {"type": "dict", "children": {str(k): describe_value(v) for k, v in value.items()}}
    if isinstance(value, list):
        if not value:
            return {"type": "list", "item": {"type": "unknown"}}
        return {"type": "list", "item": describe_value(value[0])}
    return {"type": "value", "value_type": type(value).__name__}


def first_non_empty_json_line(file_path: str) -> Optional[Any]:
    import json as _json

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                return _json.loads(line)
            except Exception:
                continue
    return None


def list_matching_files(path_pattern: str) -> List[str]:
    if not os.path.isabs(path_pattern):
        raise HTTPException(status_code=400, detail="文件路径必须是绝对路径")
    return sorted(glob.glob(path_pattern, recursive=True))


def pick_final_result_path(seq: OperationSequence) -> Optional[str]:
    for node in reversed(seq.nodes or []):
        if node.node_type == "op" and node.result_save_path:
            return node.result_save_path
    return seq.result_save_path


# ======================
# 占位符解析（删冗余版）
# 统一入口：resolve_refs_in_value()
# 支持两种引用形式：
#   1) token: {{var}} / {{fg:<id or name>}} / {{node_out:<id or name>}} / {{node_output:...}}
#   2) ref object: {"$ref": {"kind":"file_group"/"node_output", ...}}
# ======================

TOKEN_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")
FULL_TOKEN_PATTERN = re.compile(r"^\{\{([^{}]+)\}\}$")


def build_vars_context(
    task: Optional[Task] = None,
    sequence: Optional[OperationSequence] = None,
    file_group: Optional[FileGroup] = None,
    node: Optional[OperationNode] = None,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}

    if task is not None:
        if isinstance(task.vars, dict):
            ctx.update(task.vars)
        ctx.update({"task_id": task.id, "task_name": task.name})

    if sequence is not None:
        ctx.update(
            {
                "sequence_id": sequence.id,
                "sequence_name": sequence.name,
                "file_group_id": sequence.file_group_id,
            }
        )

    if file_group is not None:
        ctx.update({"file_group_id": file_group.id, "file_group_name": file_group.name})

    if node is not None:
        ctx.update({"node_id": node.id, "node_name": node.name})

    # ctx.update({"base_dir": str(BASE_DIR), "storage_dir": str(STORAGE_DIR)})
    return ctx


def _split_token(token: str) -> Tuple[str, str]:
    token = (token or "").strip()
    if ":" in token:
        k, a = token.split(":", 1)
        return k.strip().lower(), a.strip()
    return token.strip(), ""


def _find_file_group_by_id_or_name(
    state: AppState,
    task: Optional[Task],
    fg_id: str,
    fg_name: str,
) -> Optional[FileGroup]:
    if fg_id and fg_id in state.file_groups:
        return state.file_groups[fg_id]
    if task and fg_name:
        cands = [x for x in state.file_groups.values() if x.task_id == task.id and x.name == fg_name]
        if len(cands) == 1:
            return cands[0]
    return None


def _find_node_by_id_or_name(
    seq: Optional[OperationSequence],
    node_id: str,
    node_name: str,
) -> Optional[OperationNode]:
    if not seq:
        return None
    if node_id:
        for n in (seq.nodes or []):
            if n.id == node_id:
                return n
    if node_name:
        cands = [n for n in (seq.nodes or []) if n.name == node_name]
        if len(cands) == 1:
            return cands[0]
    return None


def _token_to_ref(token: str) -> Optional[Dict[str, Any]]:
    """
    把 token 的“引用类语义”统一翻译成 $ref，然后复用 resolve_ref_object()
    - {{fg:xxx}} 支持：先按 id 找，找不到再按 name（限定当前 task）找
    - {{node_out:xxx}} 支持：先按 id 找，找不到再按 name（限定当前 sequence）找
    """
    key, arg = _split_token(token)
    if not arg:
        return None

    if key in {"fg", "file_group", "filegroup"}:
        # arg 可能是 file_group_id，也可能是 file_group_name
        return {"kind": "file_group", "file_group_id": arg, "file_group_name": arg}

    if key in {"node_out", "node_output", "node", "out"}:
        return {"kind": "node_output", "node_id": arg, "node_name": arg}

    return None


def resolve_ref_object(
    ref: Dict[str, Any],
    state: AppState,
    task: Optional[Task],
    sequence: Optional[OperationSequence],
    file_group: Optional[FileGroup],
    node: Optional[OperationNode],
    ctx: Dict[str, Any],
    _stack: set,
) -> Any:
    """
    支持：
      kind=file_group  -> 返回 list[str]
      kind=node_output -> 返回 str
    """
    kind = (ref.get("kind") or "").strip()

    if kind == "file_group":
        fg = _find_file_group_by_id_or_name(
            state,
            task,
            fg_id=str(ref.get("file_group_id") or ""),
            fg_name=str(ref.get("file_group_name") or ""),
        )
        if not fg:
            raise HTTPException(status_code=400, detail=f"$ref(file_group) 无法解析：{ref}")

        out: List[str] = []
        for fp in (fg.file_paths or []):
            # 注意：这里解析 fp.path 时，file_group 上下文用 fg
            ctx_fp = build_vars_context(task=task, sequence=sequence, file_group=fg, node=node)
            v = resolve_refs_in_value(fp.path, state, task, sequence, fg, node, ctx=ctx_fp, _stack=_stack)
            out.append(str(v))
        return out

    if kind == "node_output":
        # 可跨 sequence；若 ref 未给 sequence_id，默认使用当前 sequence
        seq_id = str(ref.get("sequence_id") or (sequence.id if sequence else ""))
        target_seq = state.operation_sequences.get(seq_id) if seq_id else sequence
        if not target_seq:
            raise HTTPException(status_code=400, detail=f"$ref(node_output) 找不到 sequence：{ref}")

        target_node = _find_node_by_id_or_name(
            target_seq,
            node_id=str(ref.get("node_id") or ""),
            node_name=str(ref.get("node_name") or ""),
        )
        if not target_node:
            raise HTTPException(status_code=400, detail=f"$ref(node_output) 找不到 node：{ref}")

        if not target_node.result_save_path:
            raise HTTPException(status_code=400, detail=f"$ref(node_output) 目标节点未配置 result_save_path：{ref}")

        fg2 = state.file_groups.get(target_seq.file_group_id) if getattr(target_seq, "file_group_id", None) else None
        ctx2 = build_vars_context(task=task, sequence=target_seq, file_group=fg2, node=target_node)
        return resolve_refs_in_value(
            target_node.result_save_path, state, task, target_seq, fg2, target_node, ctx=ctx2, _stack=_stack
        )

    raise HTTPException(status_code=400, detail=f"未知 $ref.kind：{kind}")


def resolve_refs_in_string(
    s: str,
    state: AppState,
    task: Optional[Task],
    sequence: Optional[OperationSequence],
    file_group: Optional[FileGroup],
    node: Optional[OperationNode],
    ctx: Dict[str, Any],
    _stack: set,
) -> Any:
    """
    解析字符串中的 {{...}}：
    - 若整个字符串就是一个 token（比如 "{{fg:xxx}}"），返回 token 的真实值（可保留类型 list/dict/int/bool）
    - 否则：把 token 替换成 str（list/dict 会 JSON 化）
    """
    if not isinstance(s, str) or not s:
        return s

    m_full = FULL_TOKEN_PATTERN.match(s.strip())
    if m_full:
        token = m_full.group(1).strip()

        # 1) 先把引用 token 翻译成 $ref 走统一解析
        ref = _token_to_ref(token)
        if ref is not None:
            sig = json.dumps(ref, ensure_ascii=False, sort_keys=True)
            if sig in _stack:
                raise HTTPException(status_code=400, detail=f"$ref/token 循环引用：{ref}")
            _stack.add(sig)
            try:
                return resolve_ref_object(ref, state, task, sequence, file_group, node, ctx=ctx, _stack=_stack)
            finally:
                _stack.remove(sig)

        # 2) 普通变量 {{var}}
        if token in ctx and ctx[token] is not None:
            return ctx[token]

        return f"{{{{{token}}}}}"

    def _repl(m):
        token = m.group(1).strip()

        ref = _token_to_ref(token)
        if ref is not None:
            sig = json.dumps(ref, ensure_ascii=False, sort_keys=True)
            if sig in _stack:
                raise HTTPException(status_code=400, detail=f"$ref/token 循环引用：{ref}")
            _stack.add(sig)
            try:
                v = resolve_ref_object(ref, state, task, sequence, file_group, node, ctx=ctx, _stack=_stack)
            finally:
                _stack.remove(sig)
        elif token in ctx and ctx[token] is not None:
            v = ctx[token]
        else:
            v = f"{{{{{token}}}}}"

        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return "" if v is None else str(v)

    return TOKEN_PATTERN.sub(_repl, s)


def resolve_refs_in_value(
    obj: Any,
    state: AppState,
    task: Optional[Task],
    sequence: Optional[OperationSequence],
    file_group: Optional[FileGroup],
    node: Optional[OperationNode],
    ctx: Optional[Dict[str, Any]] = None,
    _stack: Optional[set] = None,
) -> Any:
    """
    深度递归替换：
    - str: {{...}} token
    - dict: 支持 {"$ref": {...}}，否则递归 value
    - list: 逐项递归
    """
    if _stack is None:
        _stack = set()
    if ctx is None:
        ctx = build_vars_context(task=task, sequence=sequence, file_group=file_group, node=node)

    if isinstance(obj, str):
        return resolve_refs_in_string(obj, state, task, sequence, file_group, node, ctx=ctx, _stack=_stack)

    if isinstance(obj, list):
        return [resolve_refs_in_value(x, state, task, sequence, file_group, node, ctx=ctx, _stack=_stack) for x in obj]

    if isinstance(obj, dict):
        # 优先识别 $ref
        if "$ref" in obj and isinstance(obj["$ref"], dict) and (obj["$ref"].get("kind")):
            sig = json.dumps(obj["$ref"], ensure_ascii=False, sort_keys=True)
            if sig in _stack:
                raise HTTPException(status_code=400, detail=f"$ref 循环引用：{obj['$ref']}")
            _stack.add(sig)
            try:
                return resolve_ref_object(obj["$ref"], state, task, sequence, file_group, node, ctx=ctx, _stack=_stack)
            finally:
                _stack.remove(sig)

        out = {}
        for k, v in obj.items():
            out[k] = resolve_refs_in_value(v, state, task, sequence, file_group, node, ctx=ctx, _stack=_stack)
        return out

    return obj


def _resolve_context(
    task_id: Optional[str],
    sequence_id: Optional[str],
    node_id: Optional[str],
) -> Tuple[AppState, Optional[Task], Optional[OperationSequence], Optional[FileGroup], Optional[OperationNode]]:
    state = load_state()
    task = state.tasks.get(task_id) if task_id else None
    seq = state.operation_sequences.get(sequence_id) if sequence_id else None
    fg = state.file_groups.get(seq.file_group_id) if (seq and seq.file_group_id) else None
    node = _find_node_by_id_or_name(seq, node_id=node_id or "", node_name="") if node_id else None
    return state, task, seq, fg, node


def resolve_with_context(
    obj: Any,
    task_id: Optional[str],
    sequence_id: Optional[str],
    node_id: Optional[str],
) -> Any:
    state, task, seq, fg, node = _resolve_context(task_id, sequence_id, node_id)
    ctx = build_vars_context(task=task, sequence=seq, file_group=fg, node=node)
    return resolve_refs_in_value(obj, state, task, seq, fg, node, ctx=ctx)


def _matches_from_resolved_path(resolved: Any) -> List[str]:
    """
    resolved 可以是：
      - str：一个 glob pattern（绝对路径）
      - list[str]：多个 pattern
    """
    matches: List[str] = []
    if isinstance(resolved, list):
        for pat in resolved:
            matches.extend(list_matching_files(str(pat)))
    else:
        matches = list_matching_files(str(resolved))
    return sorted(set(matches))


# ======================
# FastAPI 应用
# ======================

app = FastAPI(title="JSONL 处理工具")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================
# 前端页面
# ================


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_file = BASE_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse(
            content="<h1>index.html 未找到，请放在与 main.py 同一目录下</h1>",
            status_code=404,
        )
    return HTMLResponse(index_file.read_text("utf-8"))


# ================
# 状态读写 API
# ================


@app.get("/api/state", response_model=AppState)
def get_state() -> AppState:
    global APP_STATE
    APP_STATE = load_state()
    return APP_STATE


@app.post("/api/state")
def update_state(new_state: AppState) -> Dict[str, Any]:
    global APP_STATE

    with _STATE_LOCK:
        current = load_state()

        if new_state.version != current.version:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "state_version_conflict",
                    "server_version": current.version,
                    "client_version": new_state.version,
                },
            )

        new_state.version = current.version + 1

        if "root" not in new_state.directories:
            new_state.directories["root"] = Directory(id="root", name="根目录", parent_id=None, collapsed=False)

        _patch_sequences_parent_id(new_state)

        save_state(new_state)
        APP_STATE = new_state

    return {"success": True, "version": new_state.version}


# ================
# 文件结构 / 信息 API（支持 {{...}} 与 $ref）
# ================


@app.post("/api/file_structure", response_model=FileStructureResponse)
def api_file_structure(req: FilePathRequest) -> FileStructureResponse:
    try:
        resolved = resolve_with_context(req.path, req.task_id, req.sequence_id, req.node_id)
        matches = _matches_from_resolved_path(resolved)
    except HTTPException as e:
        return FileStructureResponse(success=False, structure=None, error=str(e.detail))
    except Exception as e:
        return FileStructureResponse(success=False, structure=None, error=str(e))

    if not matches:
        return FileStructureResponse(success=False, structure=None, error="没有匹配到任何文件")

    obj = first_non_empty_json_line(matches[0])
    if obj is None:
        return FileStructureResponse(success=False, structure=None, error="未找到非空且合法的 JSON 行")

    struct = describe_value(obj)
    root_struct = {"type": "dict", "children": struct.get("children", {})} if struct.get("type") == "dict" else struct
    return FileStructureResponse(success=True, structure=root_struct, error=None)


@app.post("/api/file_info", response_model=FileInfoResponse)
def api_file_info(req: FilePathRequest) -> FileInfoResponse:
    try:
        resolved = resolve_with_context(req.path, req.task_id, req.sequence_id, req.node_id)
        matches = _matches_from_resolved_path(resolved)
    except HTTPException as e:
        return FileInfoResponse(success=False, error=str(e.detail))
    except Exception as e:
        return FileInfoResponse(success=False, error=str(e))

    if not matches:
        return FileInfoResponse(success=False, error="没有匹配到任何文件")

    total_lines = 0
    for fp in matches:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for _ in f:
                    total_lines += 1
        except Exception:
            continue

    return FileInfoResponse(success=True, file_count=len(matches), total_lines=total_lines, error=None)


@app.post("/api/file_head_tail", response_model=HeadTailResponse)
def api_file_head_tail(req: FilePathRequest) -> HeadTailResponse:
    try:
        resolved = resolve_with_context(req.path, req.task_id, req.sequence_id, req.node_id)
        matches = _matches_from_resolved_path(resolved)
    except HTTPException as e:
        return HeadTailResponse(success=False, error=str(e.detail))
    except Exception as e:
        return HeadTailResponse(success=False, error=str(e))

    if not matches:
        return HeadTailResponse(success=False, error="没有匹配到任何文件")

    first_line: Optional[str] = None
    for fp in matches:
        first_line = first_non_empty_line_in_file(fp)
        if first_line is not None:
            break

    last_line: Optional[str] = None
    for fp in reversed(matches):
        last_line = last_non_empty_line_in_file(fp)
        if last_line is not None:
            break

    if first_line is None and last_line is None:
        return HeadTailResponse(success=False, error="所有匹配文件都没有非空行")

    return HeadTailResponse(success=True, first_line=first_line, last_line=last_line, error=None)


@app.post("/api/path_exists", response_model=PathExistsResponse)
def api_path_exists(req: PathExistsRequest) -> PathExistsResponse:
    """
    前端轮询判断某个结果路径是否存在：
    - 支持 path 中的 {{var}} / {{fg:...}} / {{node_out:...}}
    - 支持 {"$ref": {...}}（如果前端也想这么传）
    """
    try:
        resolved = resolve_with_context(req.path, req.task_id, req.sequence_id, req.node_id)
        matches = _matches_from_resolved_path(resolved)
    except HTTPException as e:
        return PathExistsResponse(success=False, exists=False, error=str(e.detail))
    except Exception as e:
        return PathExistsResponse(success=False, exists=False, error=str(e))

    return PathExistsResponse(success=True, exists=bool(matches), error=None)


# ================
# 操作序列：代码生成 / 执行接口（树形分支）
# ================


@app.post("/api/generate_code")
def api_generate_code(req: SequenceActionRequest) -> Dict[str, Any]:
    cfg = get_task_config(req.task_id, req.sequence_id)
    state: AppState = cfg["state"]
    task: Task = cfg["task"]
    sequence: OperationSequence = cfg["sequence"]
    root_group: FileGroup = cfg["sequence_group"]

    if sequence is None:
        raise HTTPException(status_code=404, detail="操作序列不存在")
    if root_group is None:
        raise HTTPException(status_code=404, detail="该操作序列绑定的文件组不存在")

    # 代码保存路径：允许 {{...}} / $ref
    code_save_path = sequence.code_save_path
    if code_save_path:
        ctx_global = build_vars_context(task=task, sequence=sequence, file_group=root_group, node=None)
        resolved_code_save_path = resolve_refs_in_value(code_save_path, state, task, sequence, root_group, None, ctx=ctx_global)
        code_save_path = pathlib.Path(str(resolved_code_save_path)).as_posix()

    if not code_save_path:
        raise HTTPException(status_code=400, detail="该操作序列未配置代码保存路径，请在画板标题下方填写后再生成代码。")

    # 1) 文件组路径：允许 {{...}} / $ref（最终写成 list[str]）
    root_path_patterns_resolved: List[str] = []
    ctx_global = build_vars_context(task=task, sequence=sequence, file_group=root_group, node=None)
    for x in root_group.file_paths or []:
        v = resolve_refs_in_value(x.path, state, task, sequence, root_group, None, ctx=ctx_global)
        root_path_patterns_resolved.append(str(v))
    root_path_patterns_json = json.dumps(root_path_patterns_resolved, ensure_ascii=False)

    nodes: List[OperationNode] = sequence.nodes or []
    if not nodes:
        raise HTTPException(status_code=400, detail="该操作序列没有任何节点")

    # 2) 收集函数定义
    builtin_funcs = {
        "delete_fields",
        "rename_fields",
        "has_key_path",
        "remove_duplicates_interior",
        "remove_duplicates_exterior",
    }
    func_defs: Dict[str, str] = {}
    for node in nodes:
        if node.node_type != "op":
            continue
        func_name = node.function_name
        if not func_name:
            continue

        if node.description and node.description.strip():
            func_defs[func_name] = node.description.rstrip() + "\n"
        else:
            if func_name not in builtin_funcs and func_name not in func_defs:
                func_defs[func_name] = (
                    f"def {func_name}(samples, **params):\n"
                    f"    raise NotImplementedError('请在节点描述中填写函数实现')\n"
                )

    # 3) 找叶子节点（只看未 frozen 的活节点）
    active_nodes = [n for n in nodes if not getattr(n, "frozen", False)]
    active_parent_ids = set()
    for n in active_nodes:
        pid = getattr(n, "parent_id", None)
        if pid:
            active_parent_ids.add(pid)

    leaf_ids: List[str] = [n.id for n in active_nodes if n.node_type == "op" and n.id not in active_parent_ids]
    if not leaf_ids:
        for n in reversed(nodes):
            if n.node_type == "op":
                leaf_ids = [n.id]
                break

    # 根节点 id
    root_id: Optional[str] = None
    for n in nodes:
        if n.node_type == "root":
            root_id = n.id
            break

    # 4) 节点 meta：params_json 深度解析（支持 {{...}} 与 $ref），并解析 result_save_path
    node_meta: Dict[str, Any] = {}
    for node in nodes:
        try:
            params_dict = json.loads(node.params_json or "{}")
            if not isinstance(params_dict, dict):
                params_dict = {}
        except Exception:
            params_dict = {}

        ctx_node = build_vars_context(task=task, sequence=sequence, file_group=root_group, node=node)
        params_dict = resolve_refs_in_value(params_dict, state, task, sequence, root_group, node, ctx=ctx_node)

        resolved_result_path = ""
        if node.result_save_path:
            resolved_result_path = str(resolve_refs_in_value(node.result_save_path, state, task, sequence, root_group, node, ctx=ctx_node))

        node_meta[node.id] = {
            "id": node.id,
            "node_type": node.node_type,
            "name": node.name,
            "function_name": node.function_name,
            "params": params_dict,
            "result_save_path": resolved_result_path,
            "executed": bool(node.executed),
            "parent_id": getattr(node, "parent_id", None),
        }

    node_meta_json = json.dumps(node_meta, ensure_ascii=False)
    leaf_ids_json = json.dumps(leaf_ids, ensure_ascii=False)
    root_id_json = json.dumps(root_id, ensure_ascii=False)

    # 5) 读取模板
    if not PROCESS_FILE.exists():
        raise HTTPException(status_code=500, detail=f"模板文件 {PROCESS_FILE} 不存在，请确认 process_template.py 是否在同一目录下。")

    with PROCESS_FILE.open("r", encoding="utf-8") as reader:
        lines_py = reader.read()

    # 6) 替换模板占位符
    lines_py = lines_py.replace('"$$PYTHON_TOOL_DIR$$"', f'"{str(BASE_DIR)}"')

    # 6.1 函数定义
    func_str_parts: List[str] = []
    for func_code in func_defs.values():
        func_str_parts.append(func_code)
        func_str_parts.append("")
    lines_py = lines_py.replace('"$$FUNC_CODE$$"', "\n".join(func_str_parts))

    # 6.2 DATA_PROCESS_CODE：纯迭代器 + 树形分支
    code_str_parts: List[str] = []
    code_str_parts.append(f"INPUT_PATH_PATTERNS = json.loads({repr(root_path_patterns_json)})")
    code_str_parts.append(f"NODE_CONFIGS = json.loads({repr(node_meta_json)})")
    code_str_parts.append(f"LEAF_NODE_IDS = json.loads({repr(leaf_ids_json)})")
    code_str_parts.append(f"ROOT_NODE_ID = json.loads({repr(root_id_json)})")
    code_str_parts.append("")
    code_str_parts.append("# node_id -> 一个函数：调用时返回该节点的迭代器（会从上游整条链重新跑一遍）")
    code_str_parts.append("PIPELINES = {}")
    code_str_parts.append("")
    code_str_parts.append("def build_pipeline(node_id):")
    code_str_parts.append("    if node_id in PIPELINES:")
    code_str_parts.append("        return PIPELINES[node_id]")
    code_str_parts.append("    cfg = NODE_CONFIGS[node_id]")
    code_str_parts.append("    node_type = cfg.get('node_type')")
    code_str_parts.append("    is_executed = bool(cfg.get('executed'))")
    code_str_parts.append("    result_save_path = cfg.get('result_save_path') or ''")
    code_str_parts.append("    params = cfg.get('params') or {}")
    code_str_parts.append("    func_name = cfg.get('function_name')")
    code_str_parts.append("    parent_id = cfg.get('parent_id')")
    code_str_parts.append("")
    code_str_parts.append("    # 根节点：从原始文件读入")
    code_str_parts.append("    if node_type == 'root':")
    code_str_parts.append("        def gen_root():")
    code_str_parts.append("            data = read_jsonl(INPUT_PATH_PATTERNS)")
    code_str_parts.append("            for sample in data:")
    code_str_parts.append("                yield sample")
    code_str_parts.append("")
    code_str_parts.append("        PIPELINES[node_id] = gen_root")
    code_str_parts.append("        return gen_root")
    code_str_parts.append("")
    code_str_parts.append("    if parent_id is None:")
    code_str_parts.append("        raise ValueError(f\"逻辑错误：节点 {node_id} ({cfg.get('name')}) 非 root 节点且无 parent_id，无法构建流。\")")
    code_str_parts.append("    parent_factory = build_pipeline(parent_id)")
    code_str_parts.append("")
    code_str_parts.append("    if not func_name:")
    code_str_parts.append("        raise ValueError(f\"配置错误：节点 {node_id} ({cfg.get('name')}) 非 root 节点必须配置 function_name。\")")
    code_str_parts.append("")
    code_str_parts.append("    def gen_op():")
    code_str_parts.append("        if is_executed and result_save_path:")
    code_str_parts.append("            data = read_jsonl(result_save_path)")
    code_str_parts.append("        else:")
    code_str_parts.append("            parent_data = parent_factory()")
    code_str_parts.append("            func = globals().get(func_name)")
    code_str_parts.append("            if func is None:")
    code_str_parts.append("                raise NameError(f'函数 {func_name} 未定义 (节点: {node_id})')")
    code_str_parts.append("            data = func(parent_data, **params)")
    code_str_parts.append("            if result_save_path:")
    code_str_parts.append("                data = save_jsonl(data, result_save_path)")
    code_str_parts.append("        for sample in data:")
    code_str_parts.append("            yield sample")
    code_str_parts.append("")
    code_str_parts.append("    PIPELINES[node_id] = gen_op")
    code_str_parts.append("    return gen_op")
    code_str_parts.append("")
    code_str_parts.append("def run_node(node_id):")
    code_str_parts.append("    factory = build_pipeline(node_id)")
    code_str_parts.append("    return factory()")
    code_str_parts.append("")
    code_str_parts.append("def run_all_leaves():")
    code_str_parts.append("    for leaf_id in LEAF_NODE_IDS:")
    code_str_parts.append("        cfg = NODE_CONFIGS.get(leaf_id) or {}")
    code_str_parts.append("        if cfg.get('executed'):")
    code_str_parts.append("            continue")
    code_str_parts.append("        for _ in tqdm.tqdm(run_node(leaf_id), desc=f\"处理进度-{cfg.get('name')}\", mininterval=10.0):")
    code_str_parts.append("            pass")
    code_str_parts.append("")
    code_str_parts.append("if __name__ == '__main__':")
    code_str_parts.append("    run_all_leaves()")
    code_str_parts.append("")

    lines_py = lines_py.replace('"$$DATA_PROCESS_CODE$$"', "\n".join(code_str_parts))

    with open(code_save_path, "w", encoding="utf-8") as writer:
        writer.write(lines_py)

    return {"success": True, "message": f"代码已生成并保存到: {code_save_path}"}


@app.post("/api/run_sequence")
def api_run_sequence(req: SequenceActionRequest) -> Dict[str, Any]:
    import subprocess

    cfg = get_task_config(req.task_id, req.sequence_id)
    state: AppState = cfg["state"]
    task: Task = cfg["task"]
    sequence: OperationSequence = cfg["sequence"]
    seq_group: FileGroup = cfg["sequence_group"]

    if sequence is None:
        raise HTTPException(status_code=404, detail="操作序列不存在")

    code_save_path = sequence.code_save_path
    if not code_save_path:
        raise HTTPException(status_code=400, detail="该操作序列尚未配置代码保存路径，或尚未生成代码。")

    ctx_global = build_vars_context(task=task, sequence=sequence, file_group=seq_group, node=None)
    resolved_code_save_path = resolve_refs_in_value(code_save_path, state, task, sequence, seq_group, None, ctx=ctx_global)
    code_path = pathlib.Path(str(resolved_code_save_path)).expanduser()

    if not code_path.exists():
        raise HTTPException(status_code=404, detail=f"代码文件 {code_path} 不存在，请先调用“生成可执行代码”。")

    python_exe = os.environ.get("PYTHON_EXECUTABLE") or os.sys.executable or "python"
    cpu = pick_next_cpu()
    p = subprocess.Popen(
        ["nice", "-n", "10", "ionice", "-c2", "-n7", python_exe, str(code_path)],
        preexec_fn=make_affinity_preexec(cpu),
    )

    pid = p.pid
    aff = _get_affinity_of_pid(pid)
    print(f"[RUN] spawned pid={pid} picked_cpu={cpu} affinity={aff} code={code_path}")

    ret = p.wait()
    if ret != 0:
        raise HTTPException(status_code=500, detail=f"执行处理脚本失败，返回码 {ret}。")

    return {"success": True, "message": f"处理脚本已执行：{code_path}（picked CPU {cpu}, affinity={aff}）"}


def get_task_config(task_id: str, sequence_id: Optional[str] = None):
    """
    根据 task_id（可选再校验 sequence_id）获取该任务的完整配置：
    - task 本身（含 vars）
    - task 下所有 file_groups
    - task 下所有 operation_sequences
    - （可选）当前这条 sequence 及其绑定的 file_group
    """
    state = load_state()

    task = state.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task_file_groups = [fg for fg in state.file_groups.values() if fg.task_id == task_id]
    task_sequences = [seq for seq in state.operation_sequences.values() if seq.task_id == task_id]

    seq = None
    seq_group = None
    if sequence_id is not None:
        seq = state.operation_sequences.get(sequence_id)
        if not seq:
            raise HTTPException(status_code=404, detail=f"Sequence {sequence_id} not found")
        if seq.task_id != task_id:
            raise HTTPException(status_code=400, detail="sequence_id 不属于该 task_id")
        seq_group = state.file_groups.get(seq.file_group_id)
        if not seq_group:
            raise HTTPException(status_code=404, detail=f"FileGroup {seq.file_group_id} for sequence not found")

    return {
        "state": state,
        "task": task,
        "file_groups": task_file_groups,
        "sequences": task_sequences,
        "sequence": seq,
        "sequence_group": seq_group,
    }

EDITOR_URI_TEMPLATE = os.getenv("EDITOR_URI_TEMPLATE", "")  
# 例（本地/桌面）：vscode://file{{path}}
# 例（你自建跳转页）：https://yourhost/editor/open?path={{path}}
# 例（code-server）：（按你实际 code-server URL 规则来拼）

@app.post("/api/editor_link")
def editor_link(payload: dict = Body(...)):
    path = (payload.get("path") or "").strip()
    if not path:
        return {"success": False, "error": "path empty"}

    tpl = (EDITOR_URI_TEMPLATE or "").strip()
    if not tpl:
        return {"success": False, "error": "EDITOR_URI_TEMPLATE not set"}

    # 编码：尽量保留常见路径字符
    encoded = quote(path.replace("\\", "/"), safe="/:._-+@,()[]{} ")
    url = tpl.replace("{{path}}", encoded)
    return {"success": True, "url": url}