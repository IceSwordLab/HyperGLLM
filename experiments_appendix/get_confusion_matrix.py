import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



with open("../datasets/Families.json", "r", encoding="utf-8") as file:
    families = json.load(file)
families_name = families
families = ["<"+item+">" for item in families]
families2id = {v: i for i, v in enumerate(families)}
print(families2id)

with open("../experiments_main/test_dir_paramS48HY4/@30000/results_file.txt", "r", encoding="utf-8") as file:
    data = json.load(file)
    ground = data["ground"]
    predict = "<white>" if data["predict"] not in families else data["predict"]
ground = np.array([families2id[item] for item in ground])
predict = np.array([families2id[item] for item in predict])



# plot
cm = confusion_matrix(ground, predict)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Oranges', cbar=True, xticklabels=families_name, yticklabels=families_name)
ax = plt.gca()
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
# plt.xlabel('Predicted ID', fontsize=18)
# plt.ylabel('Ground-Truth ID', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.savefig("confusion_matrix.png", dpi=600, bbox_inches='tight', pad_inches=0)#plt.show()
plt.close()






