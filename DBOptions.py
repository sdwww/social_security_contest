import os

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.ZHS16GBK'
import cx_Oracle as db


# 连接数据库
def connect():
    con = db.connect('TIANCHI_2017', '123456', '127.0.0.1:1521/ORCL')
    return con


# 执行select语句
def getSQL(sql, cursor):
    cursor.execute(sql)
    result = cursor.fetchall()
    content = []
    for row in result:
        if len(row) == 1:
            content.append(row[0])
        else:
            content.append(list(row))
    return content


# 执行更新操作
def exeSQL(sql, cursor, con):
    cursor.execute(sql)
    con.commit()