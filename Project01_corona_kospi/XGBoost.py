import xgboost as xgb
import xgboost as XGBClassifier

#방법1
# XGBoost를 사용하기 위해서는 DMatrix 형태로 변환해 주어야 합니다
dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val, y_val)
dtest = xgb.DMatrix(x_test)

# 모델 생성
# num_boost_round 만큼 반복하는데 early_stopping_rounds 만큼 성능 향상이 없으면 중단
# early_stopping_rounds를 사용하려면 eval 데이터 셋을 명기해야함
param = {파라미터 설정}
xgb_model = xgb.train(params = params, dtrain = dtrain, num_boost_round = 400, 
                        early_stopping_rounds = 100, evals=[(dtrain,'train'),(dval,'eval')])

# 예측하기, 확률값으로 반환됨
y_pre_probs = xgb_model.predict(dtest)

# 0또는 1로 변경
y_preds = [1 if x>0.5 else 0 for x in y_pre_probs]

# 특성 중요도 시각화
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)

##방법2
# 객체 생성
model = XGBClassifier(파라미터들) # gbtree : 트리 기반 모델, gblinear : 선형 모델
xgb_model = model.fit(x_train, y_train, early_stopping_rounds=100, 
                        eval_metric='logloss',eval_set=[(X_val, y_val)])

# 예측하기
y_pre = xgb_model.predict(X_test)
y_pred_probs = xgb_model.predict_proba(X_test)[:,1]

# 특성 중요도 시각화
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)

###파라미터
- General Parameters ( XGBoost 의 어떤 모델을 쓸거야? )
▶ booster [default = 'gbtree'] 

        gbtree : 트리 기반 모델

        gblinear : 선형 모델

 ▶ silent [default = 0]

        0 : 동작 메세지 프린트 함

        1 : 동작 메세지 프린트 안함 

▶ nthread [default = 전체 다 사용]

- Booster Parameters ( 모델의 조건 설정 )
▶ n_estimators [default = 100] : 나무의 개수 (=num_boost_round [default = 10] : 파이썬 래퍼에서 적용)

 ▶ early_stopping_rounds 

        최대한 몇 개의 트리를 완성해볼 것인지 

        valid loss에 더이상 진전이 없으면 멈춤

        과적합을 방지할 수 있음, n_estimators 가 높을때 주로 사용.

 ▶ learning_rate [default = 0.1] (=eta [default = 0.3] : 파이썬 래퍼에서 적용)

        학습 단계별로 가중치를 얼만큼 사용할지 결정/ 이전의 결과를 얼마나 반영할건지

        낮은 eta -> 낮은 가중치 -> 다음 단계의 결과물 적게 반영 -> 보수적

        일반적으로 0.01 ~ 0.2

        높은 값으로 다른 파라미터 조절하여 결정한 후, 낮춰서 최적의 파라미 결정

        * gradient boost에서는 기울기의 의미, 작으면 꼼꼼히 내려가고 크면 급하게 내려감

 ▶ min_child_weight [default = 1]

        child 에서 필요한 모든 관측치에 대한 가중치의 최소 합

        이 값보다 샘플 수가 작으면 leaf node가 되는 것

        너무 크면 under-fitting 될 수 있음

        CV로 조절해야함

 ▶ max_depth [default = 6]

        트리의 최대 깊이

        일반적으로 3 ~ 10  

        CV로 조절해야함

 ▶ gamma [default = 0]

        트리에서 추가적으로 가지를 나눌지를 결정할 최소 손실 감소 값

        값이 클수록 과적합 감소 효과

 ▶ subsample [default = 1] (=sub_sample : 파이썬 래퍼에서 적용)

        각 트리마다 데이터 샘플링 비율

        over-fitting 방지

        일반적으로 0.5 ~ 1

 ▶ colsample_bytree [default = 1]

        각 트리마다 feature 샘플링 비율

        일반적으로 0.5 ~ 1

 ▶ reg_lambda [default = 1] (=lambda : 파이썬 래퍼에서 적용)

        L2 regularization(ex. 릿지) 가중치

        클수록 보수적

 ▶ reg_alpha [default = 0] (=alpha : 파이썬 래퍼에서 적용)

        L1 regularization(ex. 라쏘) 가중치

        클수록 보수적

        특성이 매우 많은때 사용해볼만 함

 ▶ scale_pos_weight [default = 1]

        데이터가 불균형할때 사용, 0보다 큰 값

        보통 값을 음성 데이터 수/ 양성 데이터 수 값으로 함

 
- Learning Task Parameters ( 모델의 목표 및 계산 방법 설정 )
▶ objective [default = reg:linear] (목적 함수)

        binary:logistic :이진 분류를 위한 로지스틱 회귀, 클래스가 아닌 예측된 확률 반환

        multi:softmax : softmax를 사용한 다중 클래스 분류, 확률이 아닌 예측된 클래스 반환

        multi:softprob : softmax와 같지만 각 클래스에 대한 예상 확률 반환

 ▶ eval_metric [목적 함수에 따라 디폴트 값이 다름(회귀-rmse / 분류-error)]

        rmse : root mean square error

        mae : mean absolute error

        logloss : negative log-likelihood

        error : binary classificaion error rate (임계값 0.5)

        merror : multiclass classification error rate

        mlogloss : multiclass logloss

        auc : area under the curve

 ▶ seed [default = 0]

        시드값 고정 (나중에 재현할때 같은 값을 출력하기 위해)
        
        
        
4. GridSearchCV
데이터가 너무 크다면 일단 트리 수를 줄이고, 기본적인 파라미터만 튜닝한 이후에 점차적으로 늘려가는 방법을 사용합니다.

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# 객체 생성, 일단은 트리 100개만 만듦
xgb_model = XGBClassifier(n_estimators=100)

# 후보 파라미터 선정
params = {'max_depth':[5,7], 'min_child_weight':[1,3], 'colsample_bytree':[0.5,0.75]}

# gridsearchcv 객체 정보 입력(어떤 모델, 파라미터 후보, 교차검증 몇 번)
gridcv = GridSearchCV(xgb_model, param_grid=params, cv=3)

# 파라미터 튜닝 시작
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='auc', eval_set=[(X_val, y_val)])

#튜닝된 파라미터 출력
print(gridcv.best_params_)

# 1차적으로 튜닝된 파라미터를 가지고 객체 생성
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=7, min_child_weight=1, colsample_bytree=0.75, reg_alpha=0.03)

# 학습
xgb_model.fit(X_train, y_train, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])
