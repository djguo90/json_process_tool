
import pandas as pd
import json
import glob


def read_xlsx(xlsx_path_patterns):
    if isinstance(xlsx_path_patterns, str):
        xlsx_path_patterns = [xlsx_path_patterns]
    for path_pattern in xlsx_path_patterns:
        path_list = sorted(glob.glob(path_pattern, recursive=True))
        for path in path_list:
            df = pd.read_excel(path)
            jd = json.loads(df.to_json(orient='records', force_ascii=False))
            for item in jd:
                yield item