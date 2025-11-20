from typing import Dict, List
from dataclasses import dataclass


@dataclass
class CouponAction:
    """クーポンアクション"""
    condition_no: int
    register_coupons: List[str]  # 登録するクーポンのリスト
    delete_coupons: List[str]    # 削除するクーポンのリスト


class CouponManager:
    """マトリックスを元にクーポンの登録・削除を管理"""

    # クーポンの種類
    COUPON_TYPES = ["一般クーポン", "モバイルクーポン", "アプリクーポン"]

    # マトリックスデータ（条件No: {登録: [...], 削除: [...]}）
    COUPON_MATRIX = {
        1: {
            "register": ["一般クーポン"],
            "delete": ["アプリクーポン"]
        },
        2: {
            "register": ["モバイルクーポン"],
            "delete": ["モバイルクーポン"]
        },
        3: {
            "register": ["アプリクーポン"],
            "delete": ["一般クーポン"]
        },
        4: {
            "register": ["一般クーポン", "モバイルクーポン"],
            "delete": ["アプリクーポン"]
        },
        5: {
            "register": ["一般クーポン", "モバイルクーポン", "アプリクーポン"],
            "delete": ["一般クーポン", "モバイルクーポン"]
        }
    }

    def __init__(self):
        """マトリックスを初期化"""
        pass

    def get_coupon_actions(self, condition_no: int) -> CouponAction:
        """
        指定された条件Noに対応するクーポンアクションを取得

        Args:
            condition_no: 条件番号（1〜5）

        Returns:
            CouponAction: 登録・削除するクーポンの情報
        """
        if condition_no not in self.COUPON_MATRIX:
            raise ValueError(f"条件No {condition_no} が見つかりません")

        condition = self.COUPON_MATRIX[condition_no]

        return CouponAction(
            condition_no=condition_no,
            register_coupons=condition["register"],
            delete_coupons=condition["delete"]
        )

    def register_coupon(self, coupon_type: str):
        """
        クーポンを登録する処理

        Args:
            coupon_type: クーポンの種類
        """
        print(f"  ✓ {coupon_type}を登録しました")
        # ここに実際の登録処理を実装
        # 例: データベースへの登録、API呼び出しなど

    def delete_coupon(self, coupon_type: str):
        """
        クーポンを削除する処理

        Args:
            coupon_type: クーポンの種類
        """
        print(f"  ✗ {coupon_type}を削除しました")
        # ここに実際の削除処理を実装
        # 例: データベースからの削除、API呼び出しなど

    def execute_condition(self, condition_no: int):
        """
        指定された条件Noに基づいてクーポンの登録・削除を実行

        Args:
            condition_no: 条件番号
        """
        print(f"\n{'='*50}")
        print(f"条件No.{condition_no} の処理を実行")
        print(f"{'='*50}")

        action = self.get_coupon_actions(condition_no)

        # 登録処理
        if action.register_coupons:
            print(f"\n【登録処理】")
            for coupon in action.register_coupons:
                self.register_coupon(coupon)
        else:
            print(f"\n【登録処理】なし")

        # 削除処理
        if action.delete_coupons:
            print(f"\n【削除処理】")
            for coupon in action.delete_coupons:
                self.delete_coupon(coupon)
        else:
            print(f"\n【削除処理】なし")

        print(f"\n条件No.{condition_no} の処理が完了しました\n")

    def show_matrix(self):
        """マトリックスの内容を表示"""
        print("\n=== クーポン条件マトリックス ===\n")
        print("条件No | 登録クーポン                      | 削除クーポン")
        print("-" * 70)

        for condition_no in sorted(self.COUPON_MATRIX.keys()):
            condition = self.COUPON_MATRIX[condition_no]
            register = ", ".join(condition["register"]) if condition["register"] else "なし"
            delete = ", ".join(condition["delete"]) if condition["delete"] else "なし"
            print(f"  {condition_no}    | {register:<30} | {delete}")
        print()


def main():
    """メイン処理"""
    import sys

    manager = CouponManager()

    # 引数がない場合は使用方法を表示
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python coupon_manager.py show         - マトリックスを表示")
        print("  python coupon_manager.py execute <条件No> - 指定条件を実行")
        print("  python coupon_manager.py all          - 全条件を実行")
        return

    command = sys.argv[1]

    if command == "show":
        # マトリックスの表示
        manager.show_matrix()

    elif command == "execute":
        # 特定の条件を実行
        if len(sys.argv) < 3:
            print("エラー: 条件Noを指定してください")
            print("例: python coupon_manager.py execute 1")
            return

        condition_no = int(sys.argv[2])
        manager.execute_condition(condition_no)

    elif command == "all":
        # 全ての条件を実行
        for condition_no in range(1, 6):
            manager.execute_condition(condition_no)

    else:
        print(f"不明なコマンド: {command}")


if __name__ == "__main__":
    main()
