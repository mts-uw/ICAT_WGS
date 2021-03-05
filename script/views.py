import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap


def one_shot_plot(feat, target, model, xylim=[0, 35],  random_state=1126, save=False):
    plt.figure()
    plt.subplot().set_aspect('equal')
    x_train, x_test, y_train, y_test = train_test_split(
        feat, target, test_size=0.1, random_state=random_state)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    plt.plot(y_test, y_test_pred, 'o', c='red',
             markersize=3, alpha=0.4, label='test')
    plt.plot(y_train, y_train_pred, 'o', c='blue',
             markersize=3, alpha=0.4, label='train')

    plt.plot([-100, 200], [-100, 200], c='0', ls='-', lw=1.0)
    plt.xlim(xylim)
    plt.ylim(xylim)
    plt.xlabel("Experimental {} [%]".format(target.name))
    plt.ylabel("Predicted {} [%]".format(target.name))

    if save is not False:
        plt.savefig(save + '_one_shot_plot.svg', dpi=100, bbox_inches='tight')


def plot_importance(model, labels, topk, save=False):
    plt.figure(figsize=(6, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    topk_idx = indices[-topk:]
    plt.barh(range(len(topk_idx)),
             importances[topk_idx], color='blue', align='center')
    plt.yticks(range(len(topk_idx)), labels[topk_idx])
    plt.ylim([-1, len(topk_idx)])
    plt.xlabel("Feature Importance")
    if save is not False:
        plt.savefig(save + '_importance.svg', dpi=100, bbox_inches='tight')


def shap_importance(model, feat, target):
    explainer = shap.TreeExplainer(model=model)
    shap_values = explainer.shap_values(feat)

    shap.summary_plot(shap_values, feat, show=False)
    if save is not False:
        plt.savefig(save + '_shap.svg', dpi=100, bbox_inches='tight')
