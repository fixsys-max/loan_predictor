import streamlit as st
import pandas as pd
from model import load_model, preprocess_data, predict_model


def process_main_page():
    show_main_page()
    process_side_bar()


def show_main_page():
    st.set_page_config(
        layout='centered',
        initial_sidebar_state='expanded',
        page_title='Прогноз одобрения кредита',
        page_icon=':money_with_wings:',
    )
    st.title('Прогноз одобрения кредита')
    st.write('Приложение позволяет предсказать одобрение кредита на основе предоставленных данных.')


def write_user_data(df: pd.DataFrame):
    st.write('Данные пользователя:')
    st.write(df)


def write_prediction_result(predicted_approval: int):
    if predicted_approval == 1:
        st.info('Результат предсказания: Одобрен')
    else:
        st.error('Результат предсказания: Отклонен')
    st.markdown(f'Результат предсказания: {":green[Одобрен]" if predicted_approval == 1 else ":red[Отклонен]"}')


def process_side_bar():
    st.sidebar.header('Ввод данных:')
    income = st.sidebar.number_input('Годовой доход', min_value=0)
    credit_score = st.sidebar.number_input('Кредитный рейтинг', min_value=0, max_value=1000)
    loan_amount = st.sidebar.number_input('Сумма кредита', min_value=0)
    years_employed = st.sidebar.number_input('Стаж работы', min_value=0)
    points = st.sidebar.number_input('Баллы фактора риска', min_value=0)

    model = load_model()

    if st.sidebar.button('Предсказать'):
        new_customer = pd.DataFrame({
            'income': [income],
            'credit_score': [credit_score],
            'loan_amount': [loan_amount],
            'years_employed': [years_employed],
            'points': [points]
        })
        new_customer = preprocess_data(new_customer)
        predicted_approval = predict_model(model, new_customer)
        write_user_data(new_customer)
        write_prediction_result(predicted_approval)


if __name__ == '__main__':
    process_main_page()