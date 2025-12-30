import glob
import json
import json_repair
from pathlib import Path
import sys
import tqdm
PYTHON_TOOL_DIR = (Path("$$PYTHON_TOOL_DIR$$")/"process_tools").as_posix()
if PYTHON_TOOL_DIR not in sys.path:
    sys.path.insert(0, PYTHON_TOOL_DIR)
from json_process_funcs import (
    delete_fields, 
    rename_fields, 
    remove_duplicates_interior, 
    remove_duplicates_exterior, 
    has_key_path, 
    get_values_by_key_path,
    read_jsonl,
    read_json,
    save_jsonl,
    save_data_to_excel_merge
)
from xlsx_process_funcs import (
    read_xlsx
)
from en_process_funcs import en_html_to_plain_text


"$$FUNC_CODE$$"
"$$DATA_PROCESS_CODE$$"
