import colorsys

import matplotlib.colors as mc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def __cmap_or_cmap_from_color(cmap: mcolors.Colormap | str, 
                              dark_or_light: str = 'dark',
                              reversed: bool = False,
                              lightness_factor: float = 1,
                              ) -> mcolors.ListedColormap:
    # Case: cmap is already a ListedColormap
    if isinstance(cmap, mcolors.ListedColormap):
        return cmap
    # Case: cmap is another type of colormap (e.g., LinearSegmentedColormap)
    elif isinstance(cmap, mcolors.Colormap | mcolors.LinearSegmentedColormap):
        if reversed:
            cmap = cmap.reversed()
        # convert to ListedColormap
        return mcolors.ListedColormap(cmap(np.linspace(0, 1, 256)))
    # Case: cmap is a matplotlib Colormap name
    elif isinstance(cmap, str) and not cmap.startswith('#'):
        cmap = plt.get_cmap(cmap)
        if reversed:
            cmap = cmap.reversed()
        return mcolors.ListedColormap(cmap(np.linspace(0, 1, 256)))
    # Case: cmap is a hex color string
    elif isinstance(cmap, str) and cmap.startswith('#'):
        # create LinearSegmentedColormap from hex string
        prefix = f"{dark_or_light}:" if dark_or_light in ['dark','light'] else dark_or_light
        cmap = __lighten_color(cmap, amount=lightness_factor)
        sns_cmap = sns.color_palette(f"{prefix}{cmap}{'_r' if reversed else ''}", as_cmap=True)
        # convert to ListedColormap
        return mcolors.ListedColormap(sns_cmap(np.linspace(0, 1, 256)))
    # Error
    else:
        raise ValueError(f"""cmap must be a matplotlib Colormap or Colormap name 
                         or a hex color string. Got {cmap} of type {type(cmap)} instead.""")


def __lighten_color(color, amount=0.5,verbose=False):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    if verbose: 
        print('lightness: ',c[1], ' to ', max(0, min(1, amount * c[1])))
    hls = (c[0], max(0, min(1, amount * c[1])), c[2])
    rgb = colorsys.hls_to_rgb(*hls)
    return mc.to_hex(rgb)  
  