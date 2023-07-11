from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from .views import *



urlpatterns = [
    path('',index, name="home"),
    path('detect_shapes',detect_shapes,name="detect_shapes"),
    path('margin',margin, name="margin"),
    path('trend',trend, name="trend"),
    path('ranging',ranging, name="ranging"),
    path('oscillator',oscillator, name="oscillator"),
    path('volatility',volatility, name="volatility"),
    path('volumn',volumn, name="volumn"),
    path('momentum',momentum, name="momentum"),
    # AI
    path('predict', predict, name='predict'),
    #Urls
    path('candlepattern', candlepattern, name='candlepattern'),
    path('simple_moving_average', sma, name='sma'),
    path('money_flow_index', mfi, name='mfi'),
    path('moving_average_convergence_divergence', macd, name='macd'),
    path('relative_strength_index', rsi, name='rsi'),
    path('bollinger_bands', bb, name='bb'),
    path('ichimuko_cloud', ichimuko, name='ichimuko'),
    path('average_directional_index', adx, name='adx'),
    path('stochastic_momentum_index', smi, name='smi'),
    path('momentum', momentum, name='momentum'),
    path('exponential_moving_average', ema, name='ema'),
    path('moving_average', moa, name='moa'),
    path('fibonachi', fibonachi, name='fibonachi'),
    path('pivot_points',pivot_points, name='pivot_points'),
    path('suppt_resistance', suppt_resistance, name='suppt_resistance'),
    path('trendlines', trendlines, name='trendlines'),
    path('gan_fan', gan_fan, name='gan_fan'),
    path('rate_of_change', roc, name='roc'),
    path('williams_%R', wpr, name='home'),
    path('commodity_channel_index', cci, name='cci'),
    path('stochastic_oscillator', sto, name='sto'),
    path('chaikin_money_flow', cmf, name='cmf'),
    path('on_balance_volume', obv, name='obv'),
    path('average_true_range', atr, name='atr'),
    path('standard_devistion', std, name='std'),
    path('donchian_hannels', dcc, name='dcc'),
    path('chaikins_volatility', chv, name='chaikins_volatility'),
    path('parabolic_sar', pbr, name='parabolic_sar'),
    path('accumulation_distribution_line', adl, name='adl'),
    ]