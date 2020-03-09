import maron
import pandas as pd
import numpy as np
import talib as ta
import scipy.optimize as sco

ot = maron.OrderType.MARKET_OPEN   # シグナルがでた翌日の始値のタイミングでオーダー

def initialize(ctx):
    # 設定
    ctx.logger.debug("initialize() called")

    ctx.configure(
      channels={          # 利用チャンネル
        "jp.stock": {
          "symbols": [
            "jp.stock.4739",
            "jp.stock.8068",
            "jp.stock.3844",
            "jp.stock.9613",
            "jp.stock.9742",
            "jp.stock.7951",
            "jp.stock.2317",
            "jp.stock.2327",
            "jp.stock.9880",
            "jp.stock.4763",
            "jp.stock.9759",
            "jp.stock.9889",
            "jp.stock.7203",
            "jp.stock.6952",
            "jp.stock.3626",
            "jp.stock.3756",
            "jp.stock.6701",
            "jp.stock.6702",
            "jp.stock.7220",
            "jp.stock.1973",
            "jp.stock.4662",
            "jp.stock.6199",
            "jp.stock.3771",
            "jp.stock.3774",
            "jp.stock.6087",
            "jp.stock.6088",
            "jp.stock.3657",
            "jp.stock.9418",
            "jp.stock.3916",
            "jp.stock.2130",
            "jp.stock.4307",
            "jp.stock.9432",
            "jp.stock.3937",
            "jp.stock.6501",
            "jp.stock.6503",
            "jp.stock.4348",
            "jp.stock.6902",
            "jp.stock.8056",
            "jp.stock.3836",
          ],
          "columns": [
            "high_price_adj",
            "low_price_adj",
            "close_price_adj"
            "volume_adj",
            ]}}) 
    ctx.period_dict = {
    "period":[]
  }

    def _BREAK_NEW_HIGH(data):

      # 欠損値を埋める
      hp = data["high_price_adj"].fillna(method="ffill")
      lp = data["low_price_adj"].fillna(method="ffill")
      cp = data["close_price_adj"].fillna(method="ffill")
     
      m5 = cp.rolling(window=5, center=False).mean()
      m75 = cp.rolling(window=75, center=False).mean()
      
      buy_sig1 = (cp > m5) & (cp.shift(1) < m5.shift(1))
      buy_sig2 = cp > m75
      buy_sig3 = (buy_sig1) & (buy_sig2)
     
      ADX = pd.DataFrame(data=0,columns=cp.columns, index=cp.index)
      ADXR = pd.DataFrame(data=0,columns=cp.columns, index=cp.index)
      DIp = pd.DataFrame(data=0,columns=cp.columns, index=cp.index)
      DIm = pd.DataFrame(data=0,columns=cp.columns, index=cp.index)
      
      for (sym,val) in cp.items():
        ADX[sym] = ta.ADX(hp[sym].values.astype(np.double), 
                                lp[sym].values.astype(np.double), 
                                cp[sym].values.astype(np.double), timeperiod=14)
        ADXR[sym] = ta.ADXR(hp[sym].values.astype(np.double), 
                                lp[sym].values.astype(np.double), 
                                cp[sym].values.astype(np.double), timeperiod=28)
        DIp[sym] = ta.PLUS_DI(hp[sym].values.astype(np.double), 
                                lp[sym].values.astype(np.double), 
                                cp[sym].values.astype(np.double), timeperiod=14)
        DIm[sym] = ta.MINUS_DI(hp[sym].values.astype(np.double), 
                                lp[sym].values.astype(np.double), 
                                cp[sym].values.astype(np.double), timeperiod=14)
  
      buy_sig4 = (DIp > DIm) & (DIp > 20)
      
      buy_sig = (buy_sig3) & (buy_sig4)
      sell_sig = lp < m75
      
      # market_sigという全て0が格納されているデータフレームを作成
      market_sig = pd.DataFrame(data=0.0, columns=hp.columns, index=hp.index)
  
      # buy_sigがTrueのとき1.0、sell_sigがTrueのとき-1.0とおく
      market_sig[buy_sig == True] = 1.0
      market_sig[sell_sig == True] = -1.0
      market_sig[(buy_sig == True) & (sell_sig == True)] = 0.0
      
      
      cp_df = data["close_price_adj"].fillna(method="ffill")
      #日毎のリターンを入れるデータフレーム
      daily_df = pd.DataFrame(data=0, index=cp_df.index, columns=cp_df.columns)
      #調整後の重みを入れるデータフレーム 
      w_posterior_df = pd.DataFrame(data=0, index=cp_df.index[245:],columns=cp_df.columns) 

      #アロケーションの間隔の設定
      period  = 75#アロケーションの間隔  #60:142.61 ,75:155.95

      #period日ごとに、最適な比率を計算し、フラグを立てる。
      for i in range(0,len(cp_df.index[245:]),period):

        #アロケーションする日にフラグを立てる
        ctx.period_dict["period"].append(cp_df.index[245+i])

        '''
        最適化をして、w_posterior_dfを取得
        '''
         #初期値の設定
        if i == 0:#1日目は、初期値を等分とする
          x0 = [1. / len(cp_df.columns)] * len(cp_df.columns)
        else:#そうでないなら、一つ前の期のウェイト比を初期値にする
          x0 = record_weight

      #目的関数に必要な定数の準備
        #リターンのデータフレームの準備
        for sym in daily_df.columns:
          daily_df[sym] = cp_df[sym].pct_change()*1000
        #分散共分散行列を作成（過去１年分のヒストリカルデータから）
        Sigma = daily_df[cp_df.index[i]:cp_df.index[245+i]].cov().as_matrix()
        #目標となるRCの比率
        rc_t = [1. / len(cp_df.columns)] * len(cp_df.columns)

      #目的関数に必要な計算の準備
        #ポートフォリオの分散を計算する関数
        def calculate_portfolio_var(w,Sigma):
          w = np.matrix(w)
          return (w*Sigma*w.T)

        #RCの計算
        def calculate_risk_contribution(w,Sigma):
          w = np.matrix(w)
          sigma_p = np.sqrt(calculate_portfolio_var(w,Sigma))
          RC_i = np.multiply(Sigma*w.T,w.T)/sigma_p
          return RC_i

      #目的関数の設定
        def risk_budget_objective(w):
          sigma_p =  np.sqrt(calculate_portfolio_var(w,Sigma))# σ_p = √σ_p^2
          risk_target = np.asmatrix(np.multiply(sigma_p,rc_t))
          RC_i = calculate_risk_contribution(w,Sigma)
          J = sum(np.square(RC_i-risk_target.T))
          return J

      #制約条件
        #constraints	制約条件	dict または dictのシーケンス
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        #bounds	上下限制約	tupleのシーケンス
        bnds = [(0, None)] * len(cp_df.columns) 

      #最適化実行
        opts = sco.minimize(risk_budget_objective, x0=x0, method='SLSQP', bounds=bnds, constraints=cons)

      #最適解の受け渡し
        #w_posterior_dfに、最適解を代入
        w_posterior_df.loc[cp_df.index[245+i]] = opts["x"]
        ctx.logger.debug(w_posterior_df)
        #ctx.logger.debug(opts["x"])
        #来期の初期値として今期の割合を格納
        record_weight = opts["x"]
    
      return {
        "ADX":ADX, 
        "ADX-R": ADXR, 
        "DIp":DIp, 
        "DIm": DIm, 
        "mvavg5": m5,
        "mvavg75" : m75,
        "market:sig": market_sig,
        "posterior weight:g2": w_posterior_df,
        }

    # シグナル登録
    ctx.regist_signal("BREAK_NEW_HIGH", _BREAK_NEW_HIGH)
    
def handle_signals(ctx, date, df_current):
  
  market_sig = df_current["market:sig"]
  done_syms = set([])
  none_syms = set([])
  
 
  for (sym, val) in ctx.portfolio.positions.items():
      returns = val["returns"]
      if returns < -0.025:
        sec = ctx.getSecurity(sym)
        sec.order(-val["amount"], comment="損切り(%f)" % returns)
        done_syms.add(sym)
      if returns > 0.04:
        sec = ctx.getSecurity(sym)
        sec.order(val["amount"], comment="買い増し(%f)" % returns)
        done_syms.add(sym)
        
  buy = market_sig[market_sig > 0.0]
  for (sym, val) in buy.items():
    if sym in done_syms:
      continue
    if date in ctx.period_dict["period"]:
      sec= ctx.getSecurity(sym)
      target_weight = df_current.loc[sym,"posterior weight:g2"]
      sec.order_target_percent(target_weight*0.9,comment="")
  pass 
