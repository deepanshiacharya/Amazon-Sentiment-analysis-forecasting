import pandas as pd
import os 

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

def sampling_from_scores(
    inferenced_file: str = "/teamspace/studios/this_studio/Final_pipeline/inferences/inference_04.parquet",
    total : int = 600,
    alpha: float = 0.6,
    beta: float = 0.0,
    output_path: str = "/teamspace/studios/this_studio/Final_pipeline/sampled_by_score/sample_05.parquet",
    path = "/teamspace/studios/this_studio/Final_pipeline/remaining/remaining_04.parquet" ):
    
    '''Samples top_k uncertain data points using existing entropy/confidence_score.'''
      
    df = pd.read_parquet(inferenced_file).copy()

    required_cols = {"rid", "text", "confidence_score", "entropy"}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    entropy_norm = normalize(df["entropy"])
    inv_confidence_norm = normalize(1 - df["confidence_score"])

    df["uncertainty_score"] =  beta * entropy_norm + (1 - beta) * inv_confidence_norm
    top_k = int(alpha * total)
    bottom_k = int((1 - alpha) * total)

    top_df = df.nlargest(top_k, "uncertainty_score").reset_index(drop=True)
    bottom_df = df.nsmallest(bottom_k, "uncertainty_score").reset_index(drop=True)
    sampled_df = pd.concat([top_df,bottom_df])
    sampled_df.to_parquet(output_path, index=False)


    sampled_rids = set(sampled_df["rid"])
    remaining_df = df[~df["rid"].isin(sampled_rids)].reset_index(drop=True)
    remaining_df = remaining_df[["rid", "text"]]


    if os.path.exists(path):
        print(f"Warning: {os.path.basename(path)} already exists and will be overwritten.")
    
    remaining_df.to_parquet(path, index=False)
    print(remaining_df.head(3))

    return sampled_df, remaining_df

inferenced_filepath = "/teamspace/studios/this_studio/Final_pipeline/inferences/inference_06.parquet"
out__path = "/teamspace/studios/this_studio/Final_pipeline/sampled_by_score/sample_07.parquet"
path = "/teamspace/studios/this_studio/Final_pipeline/remaining/remaining_06.parquet"

sampled_df, remaining_df = sampling_from_scores(inferenced_file = inferenced_filepath, total=600, alpha=0.6, beta=0.0, output_path= out__path, path=path)
