import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_file():
    # 读取数据
    df = pd.read_table("/yourfilepath/SMSSpamCollection",
                       header=None,
                       names=['label', 'sms_message'])
    # 做一个map表，0表示'ham'，1表示'spam'
    df['label'] = df.label.map({'ham': 0, 'spam': 1})

    return df

def train_and_test_data(df_data):
    # 分割训练集和测试机
    X_train, X_test, y_train, y_test = train_test_split(df_data['sms_message'],
                                                        df_data['label'],
                                                        random_state=1)
    # 创建实例
    count_vector = CountVectorizer()
    # 训练数据转成矩阵
    training_data = count_vector.fit_transform(X_train)
    # 转化测试集
    testing_data = count_vector.transform(X_test)

    naive_bayes = MultinomialNB()
    # 运用朴素贝叶斯算法
    naive_bayes.fit(training_data, y_train)
    # 预测数据
    predictions = naive_bayes.predict(testing_data)

    return predictions, X_train, X_test, y_train, y_test

def evaluate_model(predictions, y_test):
    print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
    print('Precision score: ', format(precision_score(y_test, predictions)))
    print('Recall score: ', format(recall_score(y_test, predictions)))
    print('F1 score: ', format(f1_score(y_test, predictions)))

def print_testing_result(X_test, y_test, predictions):
    category_map = {0: 'ham', 1: 'spam'}
    for message, category, real in zip(X_test[50:100], predictions[50:100], y_test[50:100]):
        print('\n recevie message:', message, '\n prediction:', category_map[category], 'true value:',
              category_map[real])

if __name__ == "__main__":
    df = read_file()

    predictions, _, X_test, _, y_test = train_and_test_data(df)

    evaluate_model(predictions, y_test)

    print_testing_result(X_test, y_test, predictions)


