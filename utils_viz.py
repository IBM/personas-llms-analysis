from upsetplot import UpSet
from upsetplot import from_contents

default_colors = [
    [92, 192, 98, 0.5],
    [90, 155, 212, 0.5],
    [246, 236, 86, 0.6],
    [241, 90, 96, 0.4],
    [255, 117, 0, 0.3],
    [82, 82, 190, 0.2],
]

default_colors = [
    [i[0] / 255.0, i[1] / 255.0, i[2] / 255.0, i[3]] for i in default_colors
]


def plot_upset_politics(politics_topics_nodes):
    """
    Upsetplots configuration for politics dimensions.

    Args:
    - politics_topics_nodes: receives a dictionary with keys of each dimension
    (e.g. 'LIBER', 'INMIG', 'LGBTQ', 'CONSE') and set of nodes as values.
    """

    sets_political = from_contents(politics_topics_nodes)

    # boiler plate code for Upset visualization
    # TODO generalize the upsetplots calls to generalize to all dimensions
    upset = UpSet(sets_political, show_counts=True)
    upset.style_subsets(
        present=["CONSE"],
        absent=["LIBER", "INMIG", "LGBTQ"],
        facecolor=default_colors[0],
    )
    upset.style_subsets(
        present=["LIBER"],
        absent=["CONSE", "INMIG", "LGBTQ"],
        facecolor=default_colors[1],
    )
    upset.style_subsets(
        present=["INMIG"],
        absent=["CONSE", "LIBER", "LGBTQ"],
        facecolor=default_colors[2],
    )
    upset.style_subsets(
        present=["LGBTQ"],
        absent=["CONSE", "LIBER", "INMIG"],
        facecolor=default_colors[3],
    )
    upset.plot()


def plot_upset_ethical(ethical_topics_nodes):
    """
    Upsetplots configuration for ethical dimensions.

    Args:
    - ethical_topics_nodes: receives a dictionary with keys of each dimension
    (e.g. 'RELAT', 'DEONT', 'NIHIL', 'UTILI', 'VIRTU') and set of nodes as values.
    """

    sets_ethical = from_contents(ethical_topics_nodes)
    upset = UpSet(sets_ethical, show_counts=True)

    # boiler plate code for Upset visualization
    upset.style_subsets(
        present=["RELAT"],
        absent=["DEONT", "NIHIL", "UTILI", "VIRTU"],
        facecolor=default_colors[0],
    )
    upset.style_subsets(
        present=["DEONT"],
        absent=["RELAT", "NIHIL", "UTILI", "VIRTU"],
        facecolor=default_colors[1],
    )
    upset.style_subsets(
        present=["NIHIL"],
        absent=["RELAT", "DEONT", "UTILI", "VIRTU"],
        facecolor=default_colors[2],
    )
    upset.style_subsets(
        present=["UTILI"],
        absent=["RELAT", "DEONT", "NIHIL", "VIRTU"],
        facecolor=default_colors[3],
    )
    upset.style_subsets(
        present=["VIRTU"],
        absent=["RELAT", "DEONT", "NIHIL", "UTILI"],
        facecolor=default_colors[4],
    )
    upset.plot()
