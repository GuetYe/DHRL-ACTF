import time


def write_log(value):
    now_time = time.time()  # 获取当前日期和时间

    time_format = '%Y-%m-%d %H:%M:%S'  # 指定日期和时间格式

    time_struct = time.localtime(now_time)  # 获取struct_time

    time_put = time.strftime(time_format, time_struct)  # 格式化时间，时间变成YYYY-MM-DD HH:MI:SS

    file_name = './log/log.log'

    log_file = open(file_name, 'a', encoding="utf-8")  # 这里用追加模式，如果文件不存在的话会自动创建

    write_value = '%s %s' % (time_put, value)

    log_file.write(write_value)

    log_file.write("\n")

    log_file.close()


if __name__ == "__main__":
    # 编写程序日志
    value = '今天将两条路径的图画好了，然后解析路径信息，并通过路由矩阵去获取整体的路由性能情况'

    write_log(value)

    print("程序日志编写完毕")
