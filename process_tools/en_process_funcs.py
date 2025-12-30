from html import unescape
import html2text
import re


def en_html_to_plain_text(html_str):
    # print(f"转换前：{html_str}")
    # 替换 <u> 标签
    html_str = html_str.replace("<u>", "underlineBegin")
    html_str = re.sub(r"<u [\s\S]*?>", "underlineBegin", html_str)
    html_str = html_str.replace("</u>", "underlineEnd")
    # 将 <br /> 替换为 \n
    html_str = html_str.replace("<br />", "\n")
    # 替换 <b> 标签
    html_str = html_str.replace("<b>", "boldBegin")
    html_str = re.sub(r"<b [\s\S]*?>", "boldBegin", html_str)
    html_str = html_str.replace("</b>", "boldEnd")

    # 使用 html2text 将 HTML 转换为纯文本
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True  # 忽略链接
    text_maker.ignore_images = True  # 忽略图片
    text_maker.ignore_emphasis = True  # 忽略强调（如斜体、粗体）
    text_maker.body_width = 0  # 不自动换行（默认 78 列自动换行，会把一段话拆成很多行）

    if hasattr(text_maker, "single_line_break"):
        text_maker.single_line_break = True

    # print(f"html_str: {html_str}")
    plain_text = text_maker.handle(html_str)
    # print(f"plain_text: {plain_text}")
    plain_text = plain_text.replace("underlineBegin", "<u>").replace("underlineEnd", "</u>")
    plain_text = plain_text.replace("boldBegin", "<b>").replace("boldEnd", "</b>")
    plain_text = re.sub(r"<u>[\s]*?</u>", "______", plain_text)
    # print(f"转换后：{plain_text}")
    if html_str == "":  # TODO 相当于增加作答区域，没验证过问题
        plain_text = "______"
        # print(f"转换后：{plain_text}")
    # 将不可见字符替换为空字符
    plain_text = re.sub(r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]', '', plain_text)
    return plain_text.strip()