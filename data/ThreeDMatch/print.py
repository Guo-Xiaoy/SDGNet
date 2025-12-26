import pickle

# 定义 .pkl 文件的路径
file_path = 'train_info.pkl'  # 将此路径替换为你的 .pkl 文件的实际路径

# 打开文件并使用 pickle 加载内容
with open(file_path, 'rb') as file:  # 'rb' 表示以二进制模式读取
    data = pickle.load(file)

# 打印加载的数据
print(data)