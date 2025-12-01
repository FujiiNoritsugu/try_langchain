"""
child_docs.jsonとtour_data.jsonを結合して、
Vector Search登録用の形式に変換するスクリプト

使用方法:
    python convert_child_docs.py
"""

import json


def convert_child_docs_for_import(
    tour_data_file: str = "tour_data.json",
    child_docs_file: str = "child_docs.json",
    output_file: str = "child_docs_import.json"
):
    """
    child_docs.jsonとtour_data.jsonを結合して、
    Vector Search登録用の形式に変換
    """
    # tour_data.jsonを読み込み
    with open(tour_data_file, 'r', encoding='utf-8') as f:
        tours = json.load(f)

    # parent_idをキーにした辞書を作成
    tours_dict = {tour['id']: tour for tour in tours}

    # child_docs.jsonを読み込み
    with open(child_docs_file, 'r', encoding='utf-8') as f:
        child_docs = json.load(f)

    # 変換後のデータ
    converted_docs = []

    for child_doc in child_docs:
        parent_id = child_doc['parent_id']

        # 親ドキュメントの情報を取得
        if parent_id not in tours_dict:
            print(f"警告: parent_id {parent_id} が tour_data.json に見つかりません")
            continue

        parent_tour = tours_dict[parent_id]

        # Vector Search登録用の形式に変換
        converted_doc = {
            "id": child_doc['id'],  # tour_001_summary などの要約ID
            "text": child_doc['content'],  # 要約テキスト（検索対象）
            "name": parent_tour['name'],
            "location": parent_tour['location'],
            "season": parent_tour['season'],
            "description": parent_tour['description'],
            "highlights": parent_tour.get('highlights', [])
        }

        converted_docs.append(converted_doc)

    # 変換後のデータを出力
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_docs, f, ensure_ascii=False, indent=2)

    print(f"変換完了!")
    print(f"  入力: {tour_data_file}, {child_docs_file}")
    print(f"  出力: {output_file}")
    print(f"  変換されたドキュメント数: {len(converted_docs)}")

    return converted_docs


if __name__ == "__main__":
    convert_child_docs_for_import()
