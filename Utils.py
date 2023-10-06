import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 

from venn import venn 

def VennForChannels(data: pd.DataFrame, channels: list, freq_cap: list=None) -> None:
    """
    channels - список солбцов в data, представляющих каналы 
    freq_cap - частота, при которую квалифицировать как "Видел", по умолчанию [1,1,1,1... ]
    """
    if len(list) > 4:
        print("Диаграмма Венна строится только до 4 каналов")
        return
    if not freq_cap:
        freq_cap = [1] * len(channels)
    assert len(freq_cap) == len(channels), "Ошибка спецификации - списки разной длины"

    venn(
        {ch: set(data[data[ch] > fc].index) for ch, fc in zip(channels, freq_cap)}, 
        fmt="{percentage:.1f}%"
    ) 

def ChannelReport(data: pd.DataFrame, channel: str, weeks: str) -> None:
    #Частотная гистограмма и охват для канала
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    data[channel].plot(kind='hist', ax=ax[0], bins=50)
    data[[weeks, channel]].groupby(weeks).apply(lambda x: (x>=1).mean()).plot(ax=ax[1])

    ax[0].set_title('Frequency historgam')
    ax[1].set_title('Coverage by week')


def MediaReport(data: pd.DataFrame, channels: list, weeks: str) -> None:
    #Общий отчет по всем каналам + диаграмма Венна
    for ch in channels:
        ChannelReport(data, ch, weeks)

    VennForChannels(data, channels)