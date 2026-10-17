import pandas as pd
import streamlit as st

from src.pipeline.inference import monthly_payment


def loan_form(defaults, key="loan"):
    def value(name, fallback, lo, hi):
        try:
            return min(hi, max(lo, float(defaults.get(name, fallback))))
        except (TypeError, ValueError):
            return float(fallback)
    with st.form(key):
        a, b, c = st.columns(3)
        with a:
            amount = st.number_input("贷款金额（元）", 1000.0, 500000.0, value("loanAmnt", 50000, 1000, 500000), step=1000.0)
            income = st.number_input("年收入（元）", 0.0, 10000000.0, value("annualIncome", 80000, 0, 10000000), step=1000.0)
            term = st.selectbox("贷款期限（年）", [3, 5], index=1 if defaults.get("term") == 5 else 0)
        with b:
            interest = st.slider("年利率（%）", 0.0, 40.0, value("interestRate", 12, 0, 40), step=0.1)
            dti = st.slider("负债收入比（%）", 0.0, 100.0, value("dti", 18, 0, 100))
            revol = st.slider("循环信用额度利用率（%）", 0.0, 150.0, value("revolUtil", 30, 0, 150))
        with c:
            fico = st.number_input("信用评分下限", 300.0, 850.0, value("ficoRangeLow", 660, 300, 850), step=5.0)
            grades = list("ABCDEFG")
            grade = st.selectbox("贷款等级", grades, index=grades.index(defaults.get("grade", "C")) if defaults.get("grade") in grades else 2)
            delinquency = st.number_input("近两年逾期次数", 0, 100, int(value("delinquency_2years", 0, 0, 100)))
        submitted = st.form_submit_button("开始风险评估", type="primary", use_container_width=True)
    row = dict(defaults)
    row.update(loanAmnt=amount, annualIncome=income, term=term, interestRate=interest,
               installment=monthly_payment(amount, interest, term), dti=dti, revolUtil=revol,
               ficoRangeLow=fico, ficoRangeHigh=min(fico+4, 850), grade=grade,
               delinquency_2years=delinquency)
    return submitted, pd.DataFrame([row])
