import pandas as pd
import os
from collections import Counter, defaultdict

top_features_counter = Counter()
bot_features_counter = Counter()
importance_sum = defaultdict(float)
importance_count = defaultdict(int)

csv_files_folder = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", "feature_importance")


for csv_file in os.listdir(csv_files_folder):
    if csv_file.endswith(".csv"):
        csv_file_path = os.path.join(csv_files_folder, csv_file)
        df = pd.read_csv(csv_file_path)
        
        for id, row in df.iterrows():
            feature = row['Feature']
            importance = row['PermutationImportance']
            importance_sum[feature] += importance
            importance_count[feature] += 1
        
        top_features = df['Feature'].head(5).tolist()
        bot_features = df['Feature'].tail(5).tolist()
        
        top_features_counter.update(top_features)
        bot_features_counter.update(bot_features)
        
average_importance = {
    feature: importance_sum[feature] / importance_count[feature]
    for feature in importance_sum
}

top_df = pd.DataFrame([
    {'Feature': feature, 'Top10_Count': count, 'Average_Importance': average_importance.get(feature, 0)}
    for feature, count in top_features_counter.items()
])
top_df = top_df.sort_values(by='Average_Importance', ascending=False)

bottom_df = pd.DataFrame([
    {'Feature': feature, 'Bottom10_Count': count, 'Average_Importance': average_importance.get(feature, 0)}
    for feature, count in bot_features_counter.items()
])
bottom_df = bottom_df.sort_values(by='Average_Importance', ascending=True)

sorted_top_importance_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", "sorted_top_feature_importance.csv")
sorted_bottom_importance_file = os.path.join("./Echolocation/FeatureExtraction/ExtractedFeatures", "sorted_bottom_feature_importance.csv")

top_df.to_csv(sorted_top_importance_file, index=False)
bottom_df.to_csv(sorted_bottom_importance_file, index=False)