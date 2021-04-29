import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def get_scores(labeled_df):
    y = labeled_df['r']
    metrics = {'Accuracy': accuracy_score,
               'Balanced accuracy': balanced_accuracy_score,
               'Precision': precision_score,
               'Recall': recall_score,
               'F1 score': f1_score,
              }
    scores = {}
    for model in ['p_1', 'p_2', 'p_3']:
        y_pred = (labeled_df[model] > 0.5).astype(int)
        model_scores = {}
        for metric, func in metrics.items():
            model_scores[metric] = func(y, y_pred)
        fpr, tpr, _ = roc_curve(y, y_pred)
        scores[model] = model_scores
        model_scores['Area under ROC curve'] = auc(fpr, tpr)
    return pd.DataFrame(scores)


def plot_probabilities():
    # app_id = '4V36A5-V2E299TKA6'
    # probabilities = {'16 of 16': [],
    #                  'at least 73 of 82': []}
    #
    # # try:
    # for p in range(1, 11):
    #     probabilities['16 of 16'].append((p/10)**16)
    #     query = 'P[X > {}] for X~B(82,{})'.format(73, (p/10))
    #     client = wolframalpha.Client(app_id)
    #     res = client.query(query)
    #     answer = next(res.results).text
    #     answer = answer.split('≈')[0]
    #     answer = list(map(int, answer.split('/')))
    #     print(answer)
    #     probabilities['at least 73 of 82'].append(answer[0]/answer[1])
    # plt.plot(probabilities)
    # plt.show()
    # # except:
    # #     print('Error! Check API key for expiry or for exceeding 2000 requests per week.')

    p1 = [x**16 for x in np.linspace(0.1, 1.0, num=25)]
    x1 = np.linspace(0.1, 1.0, num=25)
    p2 = [1.1e-42, 2.3e-21, 1e-9, 0.0101, 0.1163, 0.5632, 1]
    x2 = [0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 1]

    plt.plot(x1, p1, label='16 of 16')
    plt.plot(x2, p2, label='at least 73 of 82')
    plt.xlabel('p (probability of winning each game')
    plt.ylabel('Probabilities')
    plt.legend()
    plt.show()


def str_columns_to_one_hot(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            one_hot = pd.get_dummies(df[col], prefix=col)
            df = df.drop(col, axis=1)
            df = pd.concat([df, one_hot], axis=1)
    return df


if __name__ == "__main__":
    plot_probabilities()