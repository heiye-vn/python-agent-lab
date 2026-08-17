import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "telco.db"


def init_database():
    """初始化 SQLite 数据库并插入模拟的电信客户数据（支持中文属性）"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 创建客户基础信息表（包含 customer_name 字段）
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_info (
        customer_id VARCHAR(20) PRIMARY KEY,
        customer_name VARCHAR(50),
        gender VARCHAR(10),
        senior_citizen INTEGER,
        partner VARCHAR(10),
        dependents VARCHAR(10),
        tenure INTEGER
    );
    """)

    # 2. 创建客户服务订购表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_services (
        customer_id VARCHAR(20) PRIMARY KEY,
        phone_service VARCHAR(10),
        multiple_lines VARCHAR(20),
        internet_service VARCHAR(20),
        online_security VARCHAR(20),
        tech_support VARCHAR(20),
        streaming_tv VARCHAR(20),
        streaming_movies VARCHAR(20),
        FOREIGN KEY (customer_id) REFERENCES customer_info(customer_id)
    );
    """)

    # 3. 创建客户合约与流失表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_churn (
        customer_id VARCHAR(20) PRIMARY KEY,
        contract VARCHAR(30),
        paperless_billing VARCHAR(10),
        payment_method VARCHAR(40),
        monthly_charges REAL,
        total_charges REAL,
        churn VARCHAR(10),
        FOREIGN KEY (customer_id) REFERENCES customer_info(customer_id)
    );
    """)

    # 清空已有旧表数据以便重新初始化
    cursor.execute("DROP TABLE IF EXISTS customer_churn;")
    cursor.execute("DROP TABLE IF EXISTS customer_services;")
    cursor.execute("DROP TABLE IF EXISTS customer_info;")

    # 重新建表
    cursor.execute("""
    CREATE TABLE customer_info (
        customer_id VARCHAR(20) PRIMARY KEY,
        customer_name VARCHAR(50),
        gender VARCHAR(10),
        senior_citizen INTEGER,
        partner VARCHAR(10),
        dependents VARCHAR(10),
        tenure INTEGER
    );
    """)

    cursor.execute("""
    CREATE TABLE customer_services (
        customer_id VARCHAR(20) PRIMARY KEY,
        phone_service VARCHAR(10),
        multiple_lines VARCHAR(20),
        internet_service VARCHAR(20),
        online_security VARCHAR(20),
        tech_support VARCHAR(20),
        streaming_tv VARCHAR(20),
        streaming_movies VARCHAR(20),
        FOREIGN KEY (customer_id) REFERENCES customer_info(customer_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE customer_churn (
        customer_id VARCHAR(20) PRIMARY KEY,
        contract VARCHAR(30),
        paperless_billing VARCHAR(10),
        payment_method VARCHAR(40),
        monthly_charges REAL,
        total_charges REAL,
        churn VARCHAR(10),
        FOREIGN KEY (customer_id) REFERENCES customer_info(customer_id)
    );
    """)

    # 插入客户基础信息示例数据（20位典型客户）
    customer_info_data = [
        ("7590-VHVEG", "张雪梅", "女", 0, "是", "否", 1),
        ("5575-GNVDE", "李建国", "男", 0, "否", "否", 34),
        ("3668-QPYBK", "王海峰", "男", 0, "否", "否", 2),
        ("7795-CFOCW", "赵明德", "男", 0, "否", "否", 45),
        ("9237-HQITU", "刘丽丽", "女", 0, "否", "否", 2),
        ("9305-CDSKC", "陈小花", "女", 0, "否", "否", 8),
        ("1452-KIOVK", "杨志强", "男", 0, "否", "是", 22),
        ("6713-OKOMC", "孙晓兰", "女", 0, "否", "否", 10),
        ("7892-POOKP", "周慧敏", "女", 0, "是", "否", 28),
        ("6388-TABGU", "吴克敏", "男", 0, "否", "是", 62),
        ("9763-GRSKD", "郑文斌", "男", 0, "是", "是", 13),
        ("7469-LKBCI", "孙伟林", "男", 0, "否", "否", 16),
        ("8091-TTVAX", "朱少华", "男", 0, "是", "否", 58),
        ("0280-XJGEX", "许宏亮", "男", 0, "否", "否", 49),
        ("5129-JFDZT", "韩少波", "男", 0, "否", "否", 25),
        ("3655-SNQYZ", "冯雅洁", "女", 0, "是", "是", 69),
        ("8191-XWSZG", "蒋秀琴", "女", 0, "否", "否", 52),
        ("9959-WOFSC", "沈建林", "男", 0, "否", "否", 71),
        ("4183-MYFRB", "韩美芳", "女", 0, "否", "否", 21),
        ("8779-QRDMV", "钱德禄", "男", 1, "否", "否", 1),
    ]
    cursor.executemany(
        "INSERT INTO customer_info VALUES (?, ?, ?, ?, ?, ?, ?);",
        customer_info_data,
    )

    # 插入客户服务订购数据（全中文属性）
    customer_services_data = [
        ("7590-VHVEG", "否", "无电话服务", "DSL宽带", "否", "否", "否", "否"),
        ("5575-GNVDE", "是", "否", "DSL宽带", "是", "否", "否", "否"),
        ("3668-QPYBK", "是", "否", "DSL宽带", "是", "否", "否", "否"),
        ("7795-CFOCW", "否", "无电话服务", "DSL宽带", "是", "是", "否", "否"),
        ("9237-HQITU", "是", "否", "光纤宽带", "否", "否", "否", "否"),
        ("9305-CDSKC", "是", "是", "光纤宽带", "否", "否", "是", "是"),
        ("1452-KIOVK", "是", "是", "光纤宽带", "否", "否", "是", "否"),
        ("6713-OKOMC", "否", "无电话服务", "DSL宽带", "是", "否", "否", "否"),
        ("7892-POOKP", "是", "是", "光纤宽带", "否", "是", "是", "是"),
        ("6388-TABGU", "是", "否", "DSL宽带", "是", "否", "否", "否"),
        ("9763-GRSKD", "是", "否", "DSL宽带", "是", "否", "否", "否"),
        ("7469-LKBCI", "是", "否", "无网络服务", "无网络服务", "无网络服务", "无网络服务", "无网络服务"),
        ("8091-TTVAX", "是", "是", "光纤宽带", "否", "否", "是", "是"),
        ("0280-XJGEX", "是", "是", "光纤宽带", "否", "否", "是", "是"),
        ("5129-JFDZT", "是", "否", "光纤宽带", "是", "是", "是", "是"),
        ("3655-SNQYZ", "是", "是", "光纤宽带", "是", "是", "是", "是"),
        ("8191-XWSZG", "是", "否", "无网络服务", "无网络服务", "无网络服务", "无网络服务", "无网络服务"),
        ("9959-WOFSC", "是", "是", "DSL宽带", "是", "是", "是", "是"),
        ("4183-MYFRB", "是", "否", "光纤宽带", "否", "否", "否", "否"),
        ("8779-QRDMV", "否", "无电话服务", "DSL宽带", "否", "否", "否", "是"),
    ]
    cursor.executemany(
        "INSERT INTO customer_services VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
        customer_services_data,
    )

    # 插入客户合约与流失情况数据（全中文属性）
    customer_churn_data = [
        ("7590-VHVEG", "按月合约", "是", "电子支票", 29.85, 29.85, "否"),
        ("5575-GNVDE", "一年合约", "否", "邮寄支票", 56.95, 1889.50, "否"),
        ("3668-QPYBK", "按月合约", "是", "邮寄支票", 53.85, 108.15, "是"),
        ("7795-CFOCW", "一年合约", "否", "银行转账", 42.30, 1840.75, "否"),
        ("9237-HQITU", "按月合约", "是", "电子支票", 70.70, 151.65, "是"),
        ("9305-CDSKC", "按月合约", "是", "电子支票", 99.65, 820.50, "是"),
        ("1452-KIOVK", "按月合约", "是", "信用卡", 89.10, 1949.40, "否"),
        ("6713-OKOMC", "按月合约", "否", "邮寄支票", 29.75, 301.90, "否"),
        ("7892-POOKP", "按月合约", "是", "电子支票", 104.80, 3046.05, "是"),
        ("6388-TABGU", "一年合约", "否", "银行转账", 56.15, 3487.95, "否"),
        ("9763-GRSKD", "按月合约", "是", "邮寄支票", 49.95, 587.45, "否"),
        ("7469-LKBCI", "两年合约", "否", "信用卡", 18.95, 326.80, "否"),
        ("8091-TTVAX", "一年合约", "否", "银行转账", 100.35, 5681.10, "是"),
        ("0280-XJGEX", "按月合约", "是", "银行转账", 103.70, 5036.30, "是"),
        ("5129-JFDZT", "按月合约", "是", "电子支票", 106.70, 2682.35, "是"),
        ("3655-SNQYZ", "两年合约", "是", "信用卡", 113.25, 7895.15, "否"),
        ("8191-XWSZG", "两年合约", "否", "邮寄支票", 20.65, 1022.95, "否"),
        ("9959-WOFSC", "两年合约", "否", "银行转账", 84.80, 6152.30, "否"),
        ("4183-MYFRB", "按月合约", "是", "电子支票", 79.85, 1640.40, "否"),
        ("8779-QRDMV", "按月合约", "是", "电子支票", 39.65, 39.65, "是"),
    ]
    cursor.executemany(
        "INSERT INTO customer_churn VALUES (?, ?, ?, ?, ?, ?, ?);",
        customer_churn_data,
    )

    conn.commit()
    conn.close()
    print(f"SQLite 数据库初始化完成，文件路径: {DB_PATH}")


if __name__ == "__main__":
    init_database()
