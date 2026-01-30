import json, os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def load_library_as_df(library_path: str) -> pd.DataFrame:
    with open(library_path) as file:
        library = json.load(file)
    df = pd.DataFrame(library)
    return df

def display_library_df(library_df: pd.DataFrame) -> pd.DataFrame:
    df_display = library_df.copy()
    df_display["artist"], df_display["tags"] = df_display["artist"].apply(", ".join), df_display["tags"].apply(", ".join)
    return df_display

def count_tags(library_df: pd.DataFrame, top_n: int) -> tuple[list, list]:
    all_tags = []
    for tags in library_df["tags"]:
        all_tags.extend(tags)
    counter = Counter(all_tags)
    most_common = counter.most_common(top_n)
    labels, counts = zip(*most_common)
    total = sum(counts)
    percentages = [count/total*100 for count in counts]
    return labels, percentages

def plot_hbar(library_df: pd.DataFrame, top_n=50) -> None:
    labels, percentages = count_tags(library_df, top_n)
    plt.figure(figsize=(24,12))
    bars = plt.barh(labels, percentages, color='skyblue', edgecolor='black')
    plt.gca().invert_yaxis()
    for bar in bars:
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.3f}%', va='center')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    '''
    sns.set(style="whitegrid")
    sns.barplot(x=percentages, y=labels, palette="coolwarm")
    '''
    plt.xlabel('Percentage (%)')
    plt.title(f'Top {top_n} Tags')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(os.path.join(script_dir, "../data/"))
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, f"top_{top_n}_hbar.pdf"))


if __name__ == "__main__":
    import argparse, time, os
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--library_path', type=str, required=True, help="Path of the Music Brainz library.")
    args = parser.parse_args() 

    library_path = os.path.abspath(args.library_path)

    library_df = load_library_as_df(library_path)
    display_df = display_library_df(library_df)
    print(display_df.head())
    plot_hbar(library_df, top_n=50)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Script ran in {elapsed:.2f} seconds")
    
