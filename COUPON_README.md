# クーポン管理システム

Excelマトリックス（`matrix.xlsx`）に基づいて、クーポンの登録・削除を自動実行するシステムです。

## 機能

- Excelファイルから条件マトリックスを読み込み
- 条件Noに基づいて、該当するクーポンのみを登録・削除
- 全条件の一括実行にも対応

## マトリックス構造

| 条件No | 登録クーポン |  |  | 削除クーポン |  |  |
|--------|-------------|---|---|-------------|---|---|
|        | 一般 | モバイル | アプリ | 一般 | モバイル | アプリ |
| 1      | ○   |     |     |     |     | ○   |
| 2      |     | ○   |     |     | ○   |     |
| 3      |     |     | ○   | ○   |     |     |
| 4      | ○   | ○   |     |     |     | ○   |
| 5      | ○   | ○   | ○   | ○   | ○   |     |

## 使い方

### 1. マトリックスの表示
```bash
uv run python coupon_manager.py show
```

### 2. 特定の条件を実行
```bash
uv run python coupon_manager.py execute <条件No>
```

例:
```bash
uv run python coupon_manager.py execute 1
```

出力:
```
==================================================
条件No.1 の処理を実行
==================================================

【登録処理】
  ✓ 一般クーポンを登録しました

【削除処理】
  ✗ アプリクーポンを削除しました

条件No.1 の処理が完了しました
```

### 3. 全条件を一括実行
```bash
uv run python coupon_manager.py all
```

## コード構造

### `CouponManager` クラス

#### 主要メソッド

- **`__init__(excel_path)`**: Excelファイルを読み込んでマネージャーを初期化
- **`get_coupon_actions(condition_no)`**: 条件Noに対応するアクション（登録・削除対象）を取得
- **`register_coupon(coupon_type)`**: クーポン登録処理（カスタマイズ可能）
- **`delete_coupon(coupon_type)`**: クーポン削除処理（カスタマイズ可能）
- **`execute_condition(condition_no)`**: 条件Noに基づいて登録・削除を実行
- **`show_matrix()`**: マトリックスを表示

### プログラムでの利用例

```python
from coupon_manager import CouponManager

# マネージャーの初期化
manager = CouponManager("matrix.xlsx")

# 条件No.3を実行
manager.execute_condition(3)

# または、アクション情報だけを取得
action = manager.get_coupon_actions(3)
print(f"登録対象: {action.register_coupons}")  # ['アプリクーポン']
print(f"削除対象: {action.delete_coupons}")    # ['一般クーポン']
```

## カスタマイズ方法

実際の登録・削除処理を実装するには、以下のメソッドをカスタマイズしてください:

```python
def register_coupon(self, coupon_type: str):
    """クーポン登録処理"""
    print(f"  ✓ {coupon_type}を登録しました")

    # ここに実際の処理を追加
    # 例:
    # - データベースへのINSERT
    # - 外部APIへのPOST
    # - ファイルへの書き込み

def delete_coupon(self, coupon_type: str):
    """クーポン削除処理"""
    print(f"  ✗ {coupon_type}を削除しました")

    # ここに実際の処理を追加
    # 例:
    # - データベースからのDELETE
    # - 外部APIへのDELETE
    # - ファイルからの削除
```

## 必要なパッケージ

```bash
uv add pandas openpyxl
```

## ファイル構成

```
.
├── matrix.xlsx           # 条件マトリックス
├── coupon_manager.py     # メインプログラム
└── COUPON_README.md      # このファイル
```
