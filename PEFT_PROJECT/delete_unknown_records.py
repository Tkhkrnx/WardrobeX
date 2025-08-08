# delete_unknown_records.py
from pymilvus import connections, Collection
import time


def connect_milvus():
    """连接到Milvus数据库"""
    MILVUS_HOST = '150.158.55.76'
    MILVUS_PORT = '19530'
    COLLECTION_NAME = "fashion_features"

    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, timeout=10)
        print(f"[Milvus] 连接成功: {MILVUS_HOST}:{MILVUS_PORT}")

        # 获取集合对象
        coll = Collection(name=COLLECTION_NAME)
        coll.load()
        print(f"[Milvus] 集合 {COLLECTION_NAME} 加载成功")
        return coll
    except Exception as e:
        print(f"[Milvus] 连接失败: {e}")
        return None


def count_unknown_records(coll):
    """统计属性都为unknown的记录数量"""
    try:
        # 构建查询表达式
        expr = "color == 'unknown' and shape == 'unknown' and material == 'unknown'"

        # 查询记录数
        result = coll.query(expr=expr, output_fields=["id"])
        count = len(result)
        print(f"[统计] 属性都为unknown的记录数量: {count}")
        return count, result
    except Exception as e:
        print(f"[统计] 查询失败: {e}")
        return 0, []


def delete_unknown_records(coll, batch_size=500):
    """分批删除属性都为unknown的记录"""
    try:
        # 先统计要删除的记录数量
        count, records = count_unknown_records(coll)
        if count == 0:
            print("[删除] 没有找到属性都为unknown的记录")
            return True

        print(f"[删除] 开始删除 {count} 条属性都为unknown的记录")

        # 分批删除，使用更小的批次以避免超时
        total_deleted = 0
        batch_count = (len(records) - 1) // batch_size + 1

        for i in range(0, len(records), batch_size):
            batch_num = i // batch_size + 1
            batch = records[i:i + batch_size]
            ids_to_delete = [record['id'] for record in batch]

            print(f"[删除] 正在删除批次 {batch_num}/{batch_count}，包含 {len(ids_to_delete)} 条记录")

            # 执行删除
            try:
                delete_expr = f"id in {ids_to_delete}"
                print(f"[删除] 删除表达式: {delete_expr[:100]}{'...' if len(delete_expr) > 100 else ''}")

                start_time = time.time()
                coll.delete(expr=delete_expr)
                coll.flush()  # 确保删除操作生效
                end_time = time.time()

                print(f"[删除] 批次 {batch_num} 删除并刷新完成，耗时 {end_time - start_time:.2f} 秒")
            except Exception as delete_e:
                print(f"[删除] 批次 {batch_num} 删除失败: {delete_e}")
                # 继续处理下一个批次而不是直接返回失败
                continue

            total_deleted += len(ids_to_delete)
            print(f"[删除] 累计已删除 {total_deleted}/{count} 条记录 ({total_deleted / count * 100:.1f}%)")

            # 添加延迟避免过于频繁的操作
            time.sleep(2)

        print(f"[删除] 删除完成，共删除 {total_deleted} 条记录")
        return True
    except Exception as e:
        print(f"[删除] 删除失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("[开始] 删除Milvus中属性为unknown的数据")

    # 连接Milvus
    coll = connect_milvus()
    if not coll:
        print("[错误] 无法连接到Milvus")
        return

    try:
        # 删除属性都为unknown的记录
        print("[删除] 开始删除属性都为unknown的记录...")
        if delete_unknown_records(coll, batch_size=500):
            print("[完成] 删除操作完成")
        else:
            print("[错误] 删除操作失败")

        # 显示删除后的统计信息
        try:
            final_count, _ = count_unknown_records(coll)
            print(f"[统计] 删除后，属性都为unknown的记录剩余: {final_count}")
            total_count = coll.num_entities
            print(f"[统计] 数据库中总记录数: {total_count}")
        except Exception as e:
            print(f"[统计] 获取删除后统计信息失败: {e}")

    except Exception as e:
        print(f"[错误] 执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 断开连接
        try:
            connections.disconnect("default")
            print("[结束] Milvus连接已断开")
        except:
            pass


if __name__ == '__main__':
    main()
