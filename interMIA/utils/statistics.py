import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("data/sites/ABIDEII-KKI_1/train.csv")
val = pd.read_csv("data/sites/ABIDEII-KKI_1/val.csv")
test = pd.read_csv("data/sites/ABIDEII-KKI_1/test.csv")


def label_histo(df, title="DF"):
    one = np.count_nonzero(df.LABEL.values == 1)
    two = np.count_nonzero(df.LABEL.values == 2)

    label = ["ASD", "TC"]

    plt.bar([1, 2],[one, two], tick_label=label)
    plt.title(title)
    plt.show()

label_histo(train, "Train")
print(f"Ratio TRAIN ASD/ALL: {np.count_nonzero(train.LABEL.values == 1) / len(train)}")
label_histo(val, "Val")
print(f"Ratio VAL ASD/ALL: {np.count_nonzero(val.LABEL.values == 1) / len(val)}")
label_histo(test, "Test")
print(f"Ratio TEST ASD/ALL: {np.count_nonzero(test.LABEL.values == 1) / len(test)}")

all_one = np.count_nonzero(train.LABEL.values == 1) + np.count_nonzero(test.LABEL.values == 1) +  np.count_nonzero(val.LABEL.values == 1)
all_two = np.count_nonzero(train.LABEL.values == 2) + np.count_nonzero(test.LABEL.values == 2) +  np.count_nonzero(val.LABEL.values == 2)

print(f"Ratio ALL ASD/ALL: {all_one / (all_one + all_two)}")
