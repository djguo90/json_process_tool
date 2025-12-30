import re
import json_repair
import json
from edudp.dataflow import ChainDataFlow
from functools import partial
import regex
import copy
from workflow.two_step_substring_finder import two_step_find

'''
爬取结果解析
'''
def crawl_result_parse(data, video_field, new_field):
    if video_field not in data:
        return  True, "爬取失败", data
    
    video_str = data[video_field]
    try:
        # 解析视频讲解稿部分
        video_match = re.findall(r"<视频讲解稿>([\s\S]*?)</视频讲解稿>", video_str)
        video = video_match[0].strip() if video_match else ""
        if not video:  # 未解析出来内容
            return True, "视频讲解稿未解析出内容", data
        
        # 处理JSON包裹符
        if video.startswith("```json"):
            video = video[7:]
        if video.endswith("```"):
            video = video[:-3]
        video = video.strip()
        video = re.sub(r'}+', '}', video)  # NOTE, 这是互动视频特有的规则，不可能出现多个“}”，大幅减少解析失败的概率
        video_data = json_repair.loads(video)

        # 解析提问部分
        question_match = re.findall(r"<提问>([\s\S]*?)</提问>", video_str)
        question = question_match[0].strip() if question_match else ""
        
        # 处理JSON包裹符
        if question.startswith("```json"):
            question = question[7:]
        if question.endswith("```"):
            question = question[:-3]
        question = question.strip()
        question_data = json_repair.loads(question)
        
        # 统一提问数据格式（确保为列表）
        if isinstance(question_data, dict):
            question_data = [question_data]
        
        # 转换提问格式
        transformed_questions = []
        for q in question_data:
            transformed = {
                "step": "分析",
                "type": "出选择题",
                "index": q["id"],  # 保留原始index用于定位
                "role": "老师",
                "cont": "",
                "display_cont": {
                    "question": q["引导式问题"]["question"],
                    "options": q["引导式问题"]["options"]
                },
                "mark_cont": []  # 按需求不标画
            }
            transformed_questions.append(transformed)
        # 仅处理第一个提问（如需处理多个可删除此部分）
        if transformed_questions:
            target_index = transformed_questions[0]["index"]
            # 查找目标索引位置
            insert_pos = None
            for idx, item in enumerate(video_data):
                if int(item.get("index")) == int(target_index):
                    insert_pos = idx
                    break
            # 插入提问数据
            if insert_pos is not None:
                video_data = video_data[:insert_pos] + [transformed_questions[0]] + video_data[insert_pos:]
        
        json.dumps(video_data, ensure_ascii=False)  # 确保数据可json dumps
        data[new_field] = video_data
        return False, "", data
    
    except Exception as e:
        return True, "json解析失败", data


'''
爬取结果解析
'''
def crawl_result_parse_old(data, video_field, new_field):
    """
    最开始的爬取结果。
    """
    if video_field not in data:
        return  True, "爬取失败", data
    
    video_str = data[video_field]
    try:
        # 解析视频讲解稿部分
        video_match = re.findall(r"<视频讲解稿>([\s\S]*?)</视频讲解稿>", video_str)
        video = video_match[0].strip() if video_match else ""
        if not video:  # 未解析出来内容
            return True, "视频讲解稿未解析出内容", data
        
        # 处理JSON包裹符
        if video.startswith("```json"):
            video = video[7:]
        if video.endswith("```"):
            video = video[:-3]
        video = video.strip()
        video = re.sub(r'}+', '}', video)  # NOTE, 这是互动视频特有的规则，不可能出现多个“}”，大幅减少解析失败的概率
        video_data = json_repair.loads(video)

        # 解析提问部分
        question_match = re.findall(r"<提问>([\s\S]*?)</提问>", video_str)
        question = question_match[0].strip() if question_match else ""
        
        # 处理JSON包裹符
        if question.startswith("```json"):
            question = question[7:]
        if question.endswith("```"):
            question = question[:-3]
        question = question.strip()
        question_data = json_repair.loads(question)
        
        # 统一提问数据格式（确保为列表）
        if isinstance(question_data, dict):
            question_data = [question_data]
        
        # 转换提问格式
        transformed_questions = []
        for q in question_data:
            # print(q)
            transformed = {
                "step": "分析",
                "type": "出选择题",
                "index": q.get("index", q.get("id")),  # 保留原始index用于定位
                "role": "老师",
                "cont": "",
                "display_cont": {
                    "question": q["question"],
                    "options": q["options"]
                },
                "mark_cont": []  # 按需求不标画
            }
            transformed_questions.append(transformed)
        # 仅处理第一个提问（如需处理多个可删除此部分）
        if transformed_questions:
            target_index = transformed_questions[0]["index"]
            # 查找目标索引位置
            insert_pos = None
            for idx, item in enumerate(video_data):
                if int(item.get("index")) == int(target_index):
                    insert_pos = idx
                    break
            # 插入提问数据
            if insert_pos is not None:
                video_data = video_data[:insert_pos] + [transformed_questions[0]] + video_data[insert_pos:]
        
        json.dumps(video_data, ensure_ascii=False)  # 确保数据可json dumps
        data[new_field] = video_data
        return False, "", data
    
    except Exception as e:
        return True, "json解析失败", data


def crawl_result_parse_v0723(data, video_field, new_field):
    if video_field not in data:
        return  True, "爬取失败", data
    
    video_str = data[video_field]
    try:
        if video_str.startswith("```json"):
            video_str = video_str[7:]
        if video_str.endswith("```"):
            video_str = video_str[:-3]
        video_str = video_str.strip()
        video_str = re.sub(r'}+', '}', video_str)  # NOTE, 除了出选择题，不可能出现多个“}”，大幅减少解析失败的概率
        video_data = json_repair.loads(video_str)
        data[new_field] = video_data
        return False, "", data
    
    except Exception as e:
        return True, "json解析失败", data


'''
清理一些爬取的无用字段
'''
def crawl_result_field_clean(data):
    # 去掉爬取结果中无用的字段
    for key in ["type", "query", "query_md5", "create_time", "vip_is_get", "completion_tokens", "prompt_tokens", "update_time", "answer"]:
        if key in data:
            data.pop(key)
    return False, "", data


def format_content_check_fix(data, video_field, clear_summary_knowledge=True, check_single_summary=True, must_read_ques=True):
    """
    data: 总数据
    video_field: 字段名称
    clear_summary_knowledge: 删除总结拓展阶段的知识传授
    """
    video_list = data[video_field]
    if not isinstance(video_list, list):
        return True, "互动视频不是list", data
    for step in video_list:
        if not isinstance(step, dict):
            return True, "语义块不是字典", data
        # fix 若没有mark_cont则添加
        if "mark_cont" not in step:
            step["mark_cont"] = []
        # fix 删除可能的index和role字段
        if "index" in step:
            step.pop("index")
        if "role" in step:
            step.pop("role")
        
        if len(step) != 5:
            return True, "语义块字段数量不对", data
        if any([_ not in step for _ in ["step", "type", "cont", "mark_cont", "display_cont"]]):
            return True, "语义块字段名称不对", data
        # fix 按顺序排列各个字段
        _step = step.pop("step")
        _type = step.pop("type")
        _cont = step.pop("cont")
        _display_cont = step.pop("display_cont")
        _mark_cont = step.pop("mark_cont")
        step["step"] = _step
        step["type"] = _type
        step["cont"] = _cont
        step["display_cont"] = _display_cont
        step["mark_cont"] = _mark_cont
        
        if step["step"] not in ["读题", "分析", "答案", "总结拓展"]:
            return True, "语义块step不在枚举值范围内", data
        if step["type"] not in ["讲题", "出选择题", "知识传授"]:
            return True, "语义块type不在枚举值范围内", data
        if not isinstance(step["cont"], str):
            return True, "语义块cont不是str", data
        if not isinstance(step["mark_cont"], list):
            return True, "语义块mark_cont不是list", data
        for mc in step["mark_cont"]:
            if not isinstance(mc, str):
                return True, "语义块mark_cont中的元素不是str", data
        if step["type"] == "讲题":
            if not isinstance(step["display_cont"], str):
                return True, "讲题的display_cont不是str", data
        elif step["type"] == "知识传授":
            if not isinstance(step["display_cont"], dict):
                return True, "知识传授的display_cont不是dict", data
            if "name" not in step["display_cont"]:
                return True, "知识传授的display_cont不包含name字段", data
            if "content" not in step["display_cont"]:
                return True, "知识传授的display_cont不包含content字段", data
            if not isinstance(step["display_cont"]["name"], str):
                return True, "知识传授的display_cont的name字段不是str", data
            if not isinstance(step["display_cont"]["content"], str):
                return True, "知识传授的display_cont的content字段不是str", data
        elif step["type"] == "出选择题":
            if not isinstance(step["display_cont"], dict):
                return True, "出选择题的display_cont不是dict", data
            if "question" not in step["display_cont"]:
                return True, "出选择题的display_cont不包含question字段", data
            if "options" not in step["display_cont"]:
                return True, "出选择题的display_cont不包含options字段", data
            if not isinstance(step["display_cont"]["question"], str):
                return True, "出选择题的display_cont的question字段不是str", data
            if not isinstance(step["display_cont"]["options"], dict):
                return True, "出选择题的display_cont的options字段不是dict", data
            if list(step["display_cont"]["options"].values()) not in [["正确", "错误"], ["错误", "正确"]]:
                return True, "出选择题的display_cont的options字段的values不是正确错误或错误正确", data
        possible_type_steps = ['读题-讲题', "分析-讲题", "分析-出选择题", "分析-知识传授", "答案-讲题", "总结拓展-讲题", "总结拓展-知识传授"]
        real_type_step_list = step["step"] + "-" + step["type"]
        if real_type_step_list not in possible_type_steps:
            return True, "出现了不正确的step-type组合", data
        
        # fix: cont执行strip
        step["cont"] = step["cont"].strip()
        # fix 标画全改成mark
        step["cont"] = step["cont"].replace("<标画", "<mark").replace("</标画>", "</mark>")
        for i in range(len(step["mark_cont"])):
            is_pianduan = False
            if "<标画>" in step["mark_cont"][i] and "</标画>" in step["mark_cont"][i]:
                _mc = step["mark_cont"][i]
                _mc = _mc[_mc.index("<标画>")+len("<标画>"): _mc.index("</标画>")].strip()
                if " " not in _mc:
                    is_pianduan = True
            if is_pianduan:
                step["mark_cont"][i] = step["mark_cont"][i].replace("<标画>", "<片段-标画>").replace("</标画>", "</片段-标画>")
            else:
                step["mark_cont"][i] = step["mark_cont"][i].replace("<标画>", "<句子-标画>").replace("</标画>", "</句子-标画>")
        # fix 将出选择题cont置为空
        if step["type"] == "出选择题":
            step["cont"] = ""
        # fix 读题、出选择题、知识传授、答案、总结拓展的标画为空，同时cont中的<mark>也删掉
        if step["step"] == "读题" or step["type"] == "出选择题" or step["type"] == "知识传授" or step["step"] == "答案" or step["step"] == "总结拓展":
            step["mark_cont"] = [] 
            orig_c = step["cont"]
            step["cont"] = re.sub(r"<mark id=[0-9]*?>", "", step["cont"])
            step["cont"] = step["cont"].replace("</mark>", "")
            new_c = step["cont"]
            # if orig_c != new_c:
            #     print(orig_c)
            #     print(new_c)
            if step["type"] == "出选择题":
                orig_c = step["display_cont"]["question"]
                step["display_cont"]["question"] = re.sub(r"<mark id=[0-9]*?>", "", step["display_cont"]["question"])
                step["display_cont"]["question"] = step["display_cont"]["question"].replace("</mark>", "")
                new_c = step["display_cont"]["question"]
                if orig_c != new_c:
                    print(orig_c)
                    print(new_c)
        if step["type"] != "出选择题":
            if not step["cont"]:
                return True, "非出选择题语义块cont为空", data
    # 只保留最多一个总结拓展
    if clear_summary_knowledge:
        rm_index = []
        for i, step in enumerate(video_list):
            if step["step"] == "总结拓展":
                rm_index.append(i)
        rm_index = rm_index[1:]
        video_list =[_ for i, _ in enumerate(video_list) if i not in rm_index]
        data[video_field] = video_list
    # 仅保留一个出选择题
    # TODO 需要验证删掉出选择题不影响语义
    rm_index = []
    for i, step in enumerate(video_list):
        if step["type"] == "出选择题":
            rm_index.append(i)
    rm_index = rm_index[1:]
    video_list =[_ for i, _ in enumerate(video_list) if i not in rm_index]
    data[video_field] = video_list
    # 各语义块数量检查
    if must_read_ques:
        if len([_ for _ in video_list if _["step"] == "读题"]) != 1:
            return True, "读题环节不唯一", data
    else:
        if len([_ for _ in video_list if _["step"] == "读题"]) > 1:
            return True, "读题环节超过一个", data
    if len([_ for _ in video_list if _["step"] == "答案"]) != 1:
        return True, "答案环节不唯一", data
    if len([_ for _ in video_list if _["type"] == "出选择题"]) > 1:
        return True, "出选择题数量大于一个", data
    if len([_ for _ in video_list if _["step"] == "分析"]) == 0:
        return True, "没有分析环节", data
    if check_single_summary:
        if len([_ for _ in video_list if _["step"] == "总结拓展"]) != 1:
            return True, "总结拓展数量不是一个", data
    # 各语义块位置检查
    if must_read_ques:
        if video_list[0]["step"] != "读题":
            return True, "首个语义块不是读题", data
    if video_list[-1]["step"] not in ["总结拓展", "答案"]:
        return True, "最后一个语义块不是总结拓展", data
    # if video_list[-2]["step"] != "答案":
    #     return True, "倒数第二个语义块不是答案", data
    return False, "", data


def fix_cont(data, video_field):
    video_list = data[video_field]
    for step in video_list:
        orig_cont = step["cont"]
        cont = step['cont']
        # check项
        if step["step"] == "读题":
            words_and_chars = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', cont)
            if len(words_and_chars) > 200:
                return True, "读题过长", data

        if len(regex.findall(r'\p{Emoji_Presentation}', cont)) != 0:
            return True, "存在emoji符号", data
        
        # fix
        if step["step"] == "读题":
            ## _____(xxxx)形式数据替换为“什么”
            cont = re.sub(r"(_+)\s*（.*?）|(_+)\s*\(.*?\)", '什么', cont)
            ## 去除数字题号
            cont = re.sub(r'^\s*(?:\d+[.):\-]|\(\d+\))\s*', '', cont)
        ## 下划线替换为“什么”
        cont = re.sub(r'_+', '什么', cont)
        ## 去除可能出现的<h>和</h>
        cont = re.sub(r'<h>|</h>', '', cont)
        ## 去除不可见字符
        cont = re.sub(r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]', '', cont)
        # 去除<u>和</u>
        cont = re.sub(r"<u>([\d\s]+)</u>", r"_\1_", cont)
        cont = cont.replace("<u>", "").replace("</u>", "").replace("<b>", "").replace("</b>", "")
        # 去除多个连续的“什么”
        cont = re.sub(r"什么[ +什么]*什么|什么什么|什么 什么|什么  什么", "什么", cont)
        # 去掉开头的:和>
        cont = re.sub(r'^:', '', cont)
        cont = re.sub(r'^>', '', cont)
        # reading_chunk["cont"] = re.sub(r'<mark id=\d+>|</mark>', "", cont.strip())
        cont = re.sub(r'<underline\s*\{[^}]+\}\{([^}]+)\}>', r"\1", cont)
        cont = cont.replace("</片段-标画>", "").replace("<片段-标画>", "").replace("</句子-标画>", "").replace("<句子-标画>", "").replace("</u>", "").replace("<u>", "").replace("</h>", "").replace("<h>", "").replace("</b>", "").replace("<b>", "")
        def _repl(match):
            # 如果捕获到了保留标签，就返回原标签；否则返回空（替换掉）
            return match.group(1) if match.group(1) else ' '
        pattern = r'(<mark id=\d+>|</mark>)|[<>]'
        cont = re.sub(pattern, _repl, cont)
        step["cont"] = cont.strip()
        new_cont = step["cont"]
        # if orig_cont != new_cont:
        #     print(orig_cont)
        #     print(new_cont)
    return False, "", data


def fix_display_cont(data, video_field):
    video_list = data[video_field]
    for step in video_list:
        if step["type"] == "讲题":
            orig_cont = step["display_cont"]
            display_cont = step["display_cont"]
            display_cont = display_cont.strip()
            if step["step"] == "答案":
                display_cont = "\n".join([x.strip() for x in display_cont.split("\n")])
            # 去除不可见字符
            display_cont = re.sub(r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]', '', display_cont)
            # 去除<u>和</u>
            display_cont = re.sub(r"<u>([\d\s]+)</u>", r"_\1_", display_cont)
            display_cont = display_cont.replace("<u>", "").replace("</u>", "").replace("<b>", "").replace("</b>", "")
            # 去除可能出现的<标画>和</标画>
            display_cont = re.sub(r"<标画 id=[0-9]*?>", "", display_cont)
            display_cont = re.sub(r'<标画>|</标画>', '', display_cont)
            step["display_cont"] = display_cont
            new_cont = step["display_cont"]
        elif step["type"] == "知识传授":
            orig_cont = step["display_cont"]["content"]
            display_cont = step["display_cont"]["content"]
            display_cont = display_cont.strip()
            # 去除不可见字符
            display_cont = re.sub(r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]', '', display_cont)
            # 去除<u>和</u>
            display_cont = re.sub(r"<u>([\d\s]+)</u>", r"_\1_", display_cont)
            display_cont = display_cont.replace("<u>", "").replace("</u>", "").replace("<b>", "").replace("</b>", "")
            # 去除可能出现的<标画>和</标画>
            display_cont = re.sub(r"<标画 id=[0-9]*?>", "", display_cont)
            display_cont = re.sub(r'<标画>|</标画>', '', display_cont)
            display_cont = re.sub(r'√', '✓', display_cont)
            # display_cont = re.sub(r'×', '✗', display_cont)
            display_cont = re.sub(r'✔', '✓', display_cont)
            step["display_cont"]["content"] = display_cont
            new_cont = step["display_cont"]["content"]
            # 仅保留name和content字段
            rm_keys = [_ for _ in step["display_cont"] if _ not in ["name", "content"]]
            for k in rm_keys:
                step["display_cont"].pop(k)
        elif step["type"] == "出选择题":
            orig_cont = step["display_cont"]["question"]
            display_cont = step["display_cont"]["question"]
            display_cont = display_cont.strip()
            # 去除可能出现的<h>和</h>
            display_cont = re.sub(r'<h>|</h>', '', display_cont)
            # 去除不可见字符
            display_cont = re.sub(r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]', '', display_cont)
            # 去除<u>和</u>
            display_cont = re.sub(r"<u>([\d\s]+)</u>", r"_\1_", display_cont)
            display_cont = display_cont.replace("<u>", "").replace("</u>", "").replace("<b>", "").replace("</b>", "")
            # 去除可能出现的<标画>和</标画>
            display_cont = re.sub(r"<标画 id=[0-9]*?>", "", display_cont)
            display_cont = re.sub(r'<标画>|</标画>', '', display_cont)
            step["display_cont"]["question"] = display_cont
            new_cont = step["display_cont"]["question"]
            # 仅保留question和options字段
            rm_keys = [_ for _ in step["display_cont"] if _ not in ["question", "options"]]
            for k in rm_keys:
                step["display_cont"].pop(k)
        # if orig_cont != new_cont:
        #     print(orig_cont)
        #     print(new_cont)
    return False, "", data


def norm_str(text):
    pattern = r'[^a-zA-Z0-9\u4e00-\u9fa5]'  # 只保留大小写字母、数字、汉字
    return " ".join(re.sub(pattern, ' ', text).strip().split())


def fix_mark_cont(data, video_field, A_kuang_field):
    video_list = data[video_field]
    # A_kuang = norm_str(data[A_kuang_field])
    A_kuang = data[A_kuang_field]
    marks_record = []
    for step in video_list:
        orig_step = copy.deepcopy(step)
        is_bad_mark = False
        mark_starts = re.findall(r"<mark id=[0-9]+?>", step["cont"])  # NOTE 这里用+而不是用*
        mark_ends = re.findall(r"</mark>", step["cont"])
        if len(mark_starts) != len(mark_ends) or len(mark_starts) != len(step["mark_cont"]):
            is_bad_mark = True
        else:
            mark_starts = re.findall(r"<mark id=([0-9]+?)>", step["cont"])  # NOTE 这里用+而不是用*
            mark_starts = list(set([int(i) for i in mark_starts]))
            if len(mark_starts) != len(step["mark_cont"]) or any([i>=len(step["mark_cont"]) for i in mark_starts]):
                is_bad_mark = True
        # if any([_.count("<片段-标画>") + _.count("</片段-标画>") + _.count("<句子-标画>") + _.count("</句子-标画>")!=2 for _ in step["mark_cont"]]):
        #     is_bad_mark = True 
        # if (_.count("<片段-标画>") == 1 and _.count("</片段-标画>") == 1 and _.count("<句子-标画>") == 0 and  _.count("</句子-标画>")==0) or 
        # if any([norm_str(_.replace("<片段-标画>", "").replace("</片段-标画>", "").replace("<句子-标画>", "").replace("</句子-标画>", "")) not in A_kuang for _ in step["mark_cont"]]):
        #     is_bad_mark = True
        for mc in step["mark_cont"]:
            if not ((mc.count("<片段-标画>") == 1 and mc.count("</片段-标画>") == 1 and mc.count("<句子-标画>") == 0 and mc.count("</句子-标画>")==0) or (mc.count("<片段-标画>") == 0 and mc.count("</片段-标画>") == 0 and mc.count("<句子-标画>") == 1 and mc.count("</句子-标画>")==1)):
                is_bad_mark = True
            if "<片段-标画>" in mc and "</片段-标画>" in mc:
                mc_pos, reason = two_step_find(A_kuang, mc, start_tag="<片段-标画>", end_tag="</片段-标画>")
                if mc_pos is None:
                    is_bad_mark = True
            elif "<句子-标画>" in mc and "</句子-标画>" in mc:
                mc_pos, reason = two_step_find(A_kuang, mc, start_tag="<句子-标画>", end_tag="</句子-标画>")
                if mc_pos is None:
                    is_bad_mark = True
            

        # 记录所有标画内容，后续检查重复
        for mc in step["mark_cont"]:
            mc_clean = mc.replace("<片段-标画>", "").replace("</片段-标画>", "").replace("<句子-标画>", "").replace("</句子-标画>", "").strip()
            mc_clean = norm_str(mc_clean)
            if mc_clean in marks_record:
                is_bad_mark =True
            else:
                marks_record.append(mc_clean)
                
        if is_bad_mark:
            step["cont"] = re.sub(r"<mark id=[0-9]*?>", "", step["cont"])
            step["cont"] = re.sub(r"</mark>", "", step["cont"])
            step["mark_cont"] = []

    return False, "", data


# TODO 删掉部分简单题目的出选择题
def fix_xzt(data, video_field):
    pass

# TODO 音标题修复，需要包括中文发音示例
# 我们先来看第一个单词<mark id=0>“box”</mark>，里面的“o”发音是 /ɒ/（类似中文“奥”的短音）。 
# 接着看第二个单词<mark id=0>“soft”</mark>，里面的“o”发音也是 /ɒ/（同样类似中文“奥”的短音）。 

def fix_tts_shitiping(data, B_kuang_field, video_field, yinbiao_list, strict=True):
    def _fix(old_str, d_underline, yinbiao_list):
        old_str = old_str.replace("ˈ", "'")
        old_str = old_str.replace("</mark>", "【mark_end】")
        for i, yl in enumerate(yinbiao_list):
            old_str = old_str.replace(yl, f"音标占位符【{i}音标占位符】")  # TODO 不能替换音标中的东西
        for du in d_underline:
            old_str = re.sub(fr'(?<![a-zA-Zɪɒoʊʃɜ/-])({re.escape(du)})(?![a-zA-Zɪɒoʊʃɜ/-])', " ".join(list(du)), old_str)
        if strict:
            # old_str = re.sub(r'/([^/]*?)/', lambda m: f'/{m.group(1).replace("i", "ɪ")}/', fr'{old_str}')
            old_str = re.sub(r'/([^/]*?)/', lambda m: f'/{m.group(1).replace("ɔ", "ɒ")}/', fr'{old_str}')
            old_str = re.sub(r'/([^/]*?)/', lambda m: f'/{m.group(1).replace("ː", ":")}/', fr'{old_str}')
            old_str = re.sub(r'/([^/]*?)/', lambda m: f'/{m.group(1).replace("əʊ", "oʊ")}/', fr'{old_str}')
            old_str = old_str.replace("/hɒ:s/", "/hɔ:rs/")
            old_str = old_str.replace("/wɜ:d/", "/wɜ:rd/")
            old_str = old_str.replace("/wɜ:k/", "/wɜ:rk/")
            old_str = old_str.replace("/'weðə(r)/", "/'weðər/")
            old_str = old_str.replace("/'bɑ:skɪtbɒ:l/", "/'bæskɪtbɔ:l/")
            old_str = old_str.replace("/'lɪsn/", "/'lɪs(ə)n/")
            old_str = old_str.replace("/ʃʊd/", "/ʃʊd;ʃəd/")
            old_str = old_str.replace("/fɜ:st/", "/fɜ:rst/")
            old_str = old_str.replace("/'nu:dlz/", "/'nu:d(ə)lz/")
            old_str = old_str.replace("/drɒ:/", "/drɔ:/")
            old_str = old_str.replace("/'kwɒ:tə(r)/", "/'kwɔ:rtər/")
            old_str = old_str.replace("/'roʊbɒt/", "/'roʊbɑ:t/")
            old_str = old_str.replace("/rɪ'membə(r)/", "/rɪ'membər/")
            old_str = old_str.replace("/'bɑ:skɪt/", "/'bæskɪt/")
            old_str = old_str.replace("/fə'ɡet/", "/fər'ɡet/")
            old_str = old_str.replace("/'ɜ:lɪ/", "/'ɜ:rlɪ/")
            old_str = old_str.replace("/lɒt/", "/lɑ:t/")
            old_str = old_str.replace("/ðeə(r)/", "/ðer/")
            old_str = old_str.replace("/'blækbɒ:d/", "/'blækbɔ:rd/")
            old_str = old_str.replace("/dɪə(r)/", "/dɪr/")
            old_str = old_str.replace("/'ʌŋkl/", "/'ʌŋk(ə)l/")
            old_str = old_str.replace("/'stɒ:rɪ/", "/'stɔ:rɪ/")
            old_str = old_str.replace("/'ænɪml/", "/'ænɪm(ə)l/")
            old_str = old_str.replace("/rɒŋ/", "/rɔ:ŋ/")
            old_str = old_str.replace("/nɜ:s/", "/nɜ:rs/")
            old_str = old_str.replace("/'dɒktə(r)/", "/'dɑ:ktər/")
            old_str = old_str.replace("/wɒ:k/", "/wɔ:k/")
            old_str = old_str.replace("/'fɒ:tɪ/", "/'fɔ:rtɪ/")
        for i, yl in enumerate(yinbiao_list):
            old_str = old_str.replace(f"音标占位符【{i}音标占位符】", yl)
        old_str = re.sub(r'(?<=[a-zA-Z])/(?=[a-zA-Z])', '或者', old_str)  # 本意：he/she/it替换为he或者she或者it，TODO 50 yuan/kg
        old_str = re.sub(r'(?<![a-zA-Z])-(s)(?![a-zA-Z])', 's', old_str)  # 先移除连字符并添加空格
        old_str = re.sub(r'(?<![a-zA-Z])-(es)(?![a-zA-Z])', 'e s', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(tion)(?![a-zA-Z])', 't i o n', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(sion)(?![a-zA-Z])', 's i o n', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ment)(?![a-zA-Z])', 'm e n t', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(able)(?![a-zA-Z])', 'a b l e', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ing)(?![a-zA-Z])', 'i n g', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ed)(?![a-zA-Z])', 'e d', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(d)(?![a-zA-Z])', 'd', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ful)(?![a-zA-Z])', 'f u l', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(less)(?![a-zA-Z])', 'l e s s', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ly)(?![a-zA-Z])', 'l y', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(er)(?![a-zA-Z])', 'e r', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(or)(?![a-zA-Z])', 'o r', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(est)(?![a-zA-Z])', 'e s t', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(en)(?![a-zA-Z])', 'e n', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(al)(?![a-zA-Z])', 'a l', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ous)(?![a-zA-Z])', 'o u s', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ness)(?![a-zA-Z])', 'n e s s', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(y)(?![a-zA-Z])', 'y', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ity)(?![a-zA-Z])', 'i t y', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ship)(?![a-zA-Z])', 's h i p', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(th)(?![a-zA-Z])', 't h', old_str)
        old_str = re.sub(r'(?<![a-zA-Z])-(ty)(?![a-zA-Z])', 't y', old_str)

        # 
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(es)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'e s', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ing)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'i n g', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ed)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'e d', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(tion)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 't i o n', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(sion)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 's i o n', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ment)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'm e n t', old_str)
        # old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(able)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'a b l e', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ful)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'f u l', old_str)
        # old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(less)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'l e s s', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ly)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'l y', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(er)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'e r', old_str)
        # old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(or)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'o r', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(est)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'e s t', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(en)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'e n', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(al)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'a l', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ous)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'o u s', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ness)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'n e s s', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ity)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 'i t y', old_str)
        # old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ship)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 's h i p', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(th)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 't h', old_str)
        old_str = re.sub(r'(?<![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])(ty)(?![ɪæɜəʌɑɒɔʊʃʒθðː: /a-zA-Z])', 't y', old_str)

        old_str = old_str.replace("【mark_end】", "</mark>")
        return old_str
    d_B_kuang = data[B_kuang_field]
    d_underline = list(set(re.findall(r'[a-zA-Z]<u>(.*?)</u>', d_B_kuang) + re.findall(r'<u>(.*?)</u>[a-zA-Z]', d_B_kuang)))
    video_list = data[video_field]
    for step in video_list:
        old_str = step["cont"]
        step["cont"] = _fix(step["cont"], d_underline=d_underline, yinbiao_list=yinbiao_list)
        new_str = step['cont']
        # if old_str != new_str:
        #     print(old_str)
        #     print(new_str)
        if step["type"] == "出选择题":
            step["display_cont"]["question"] = _fix(step["display_cont"]["question"], d_underline=d_underline, yinbiao_list=yinbiao_list)
    return False, "", data


XYTextVideoCheckFlow = ChainDataFlow(
    "XY文本/暑假作业/秋季教辅-check", 
    funcs=[
        partial(crawl_result_parse, video_field="answer", new_field="video"),
        crawl_result_field_clean,
        partial(format_content_check_fix, video_field="video"),
        partial(fix_cont, video_field="video"),
        partial(fix_display_cont, video_field="video"),
        partial(fix_mark_cont, video_field="video", A_kuang_field="A_kuang"),
    ]
)


VideoReplaceCheckFlow = ChainDataFlow(
    "重刷覆盖入库check-HWL-文本", 
    funcs=[
        partial(crawl_result_parse, video_field="answer", new_field="video"),
        crawl_result_field_clean,
        partial(format_content_check_fix, video_field="video"),
        partial(fix_cont, video_field="video"),
        partial(fix_display_cont, video_field="video"),
        partial(fix_mark_cont, video_field="video", A_kuang_field="A_kuang"),
    ]
)


VideoReplaceCheckFlow2 = ChainDataFlow(
    "重刷覆盖入库check-HWL-FIG-XF", 
    funcs=[
        partial(crawl_result_parse, video_field="answer_mode4", new_field="video"),
        crawl_result_field_clean,
        partial(format_content_check_fix, video_field="video"),
        partial(fix_cont, video_field="video"),
        partial(fix_display_cont, video_field="video"),
        partial(fix_mark_cont, video_field="video", A_kuang_field="A_kuang"),
    ]
)


VideoReplaceCheckFlowJFJG = ChainDataFlow(
    "重刷覆盖入库check-JFJG", 
    funcs=[
        partial(crawl_result_parse_old, video_field="answer", new_field="video"),
        crawl_result_field_clean,
        partial(format_content_check_fix, video_field="video"),
        partial(fix_cont, video_field="video"),
        partial(fix_display_cont, video_field="video"),
        partial(fix_mark_cont, video_field="video", A_kuang_field="A_kuang"),
    ]
)


yinbiao_data_path = "/mnt3/data/djguo/辅学/T250721_音标题读音优化/all_yinbiao_in_rules.txt"
with open(yinbiao_data_path) as reader:
    yinbiao_list = [x.strip() for x in reader]


ShitipingVideoCheckFlow = ChainDataFlow(
    "试题屏互动视频检查", 
    funcs=[
        partial(crawl_result_parse_v0723, video_field="answer", new_field="video"),
        crawl_result_field_clean,
        partial(format_content_check_fix, video_field="video", clear_summary_knowledge=True, check_single_summary=False),
        partial(fix_cont, video_field="video"),
        partial(fix_display_cont, video_field="video"),
        partial(fix_mark_cont, video_field="video", A_kuang_field="A_kuang"),
        partial(fix_tts_shitiping, B_kuang_field="B_kuang", video_field="video", yinbiao_list=yinbiao_list),
    ]
)

LogicVideoCheckFlow = ChainDataFlow(
    "互动视频优化讲解逻辑检查", 
    funcs=[
        partial(crawl_result_parse_v0723, video_field="answer_mode4_2", new_field="video"),
        crawl_result_field_clean,
        partial(format_content_check_fix, video_field="video", clear_summary_knowledge=False, check_single_summary=False, must_read_ques=False),
        partial(fix_cont, video_field="video"),
        partial(fix_display_cont, video_field="video"),
        partial(fix_mark_cont, video_field="video", A_kuang_field="A_kuang"),
        partial(fix_tts_shitiping, B_kuang_field="B_kuang", video_field="video", yinbiao_list=yinbiao_list, strict=False),
    ]
)

LogicVideoTestCheckFlow = ChainDataFlow(
    "互动视频优化讲解逻辑检查-测试集", 
    funcs=[
        partial(crawl_result_parse_v0723, video_field="answer", new_field="video"),
        crawl_result_field_clean,
        partial(format_content_check_fix, video_field="video", clear_summary_knowledge=False, check_single_summary=False, must_read_ques=False),
        partial(fix_cont, video_field="video"),
        partial(fix_display_cont, video_field="video"),
        partial(fix_mark_cont, video_field="video", A_kuang_field="A_kuang"),
        partial(fix_tts_shitiping, B_kuang_field="B_kuang", video_field="video", yinbiao_list=yinbiao_list, strict=False),
    ]
)


QuesTypeOptiVideoCheckFlow = ChainDataFlow(
    "分题型优化互动视频优秀率", 
    funcs=[
        partial(crawl_result_parse_v0723, video_field="answer", new_field="video"),
        crawl_result_field_clean,
        partial(format_content_check_fix, video_field="video", clear_summary_knowledge=False, check_single_summary=False),
        partial(fix_cont, video_field="video"),
        partial(fix_display_cont, video_field="video"),
        partial(fix_mark_cont, video_field="video", A_kuang_field="A_kuang")
    ]
)