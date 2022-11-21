import matplotlib.pyplot as plt

if __name__ == "__main__":
    from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

    ##set root_dir to the correct path to your dataset folder
    root_dir = "/home/zhuhe/Dataset/data/train"

    afl = ArgoverseForecastingLoader(root_dir)

    from argoverse.visualization.visualize_sequences import viz_sequence

    which_to_vis = 100356
    seq_path = f"{root_dir}/{which_to_vis}.csv"
    viz_sequence(afl.get(seq_path).seq_df, show=False)
    plt.savefig(f"./visualize/basic_figure/"+str(which_to_vis)+".jpg")

    # file1  = open("./visualize/basic_figure/a.jpg")


