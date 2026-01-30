import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px

# load
preprocessor = joblib.load('model_data/preprocessor.joblib')
model = joblib.load('model_data/xgb_model.joblib')
features = joblib.load('model_data/features.joblib')   # list of feature names

st.title("Race Time Predictor")

u = st.file_uploader("Upload CSV with features", type=["csv"])
if u is not None:
    df = pd.read_csv(u)
    # ensure columns present
    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        X = df[features].copy()

        X_proc = preprocessor.transform(X)
        preds = model.predict(X_proc)
        df['pred_Race_Time'] = preds

        st.write(df.head())
        st.download_button("Download predictions", df.to_csv(index=False), "preds.csv")

        # If actuals present, show metrics + plots
        if 'Race_Time' in df.columns:
            mse = mean_squared_error(df['Race_Time'], df['pred_Race_Time'])
            mae = mean_absolute_error(df['Race_Time'], df['pred_Race_Time'])
            r2 = r2_score(df['Race_Time'], df['pred_Race_Time'])

            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mae:.3f}")
            c2.metric("MSE", f"{mse:.3f}")
            c3.metric("R²", f"{r2:.3f}")

            # Actual vs Predicted scatter with y=x reference
            fig = px.scatter(df, x='Race_Time', y='pred_Race_Time',
                             labels={'Race_Time': 'Actual Race Time', 'pred_Race_Time': 'Predicted Race Time'},
                             title='Actual vs Predicted Race Time',
                             hover_data=features)
            minv = float(df['Race_Time'].min())
            maxv = float(df['Race_Time'].max())
            fig.add_shape(type="line", x0=minv, x1=maxv, y0=minv, y1=maxv,
                          line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

            # Correlation heatmap for numerical features
            num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])] if len(features)>0 else []
            if len(num_cols) > 1:
                corr = df[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, title='Correlation Matrix')
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)

        # find fastest predicted lap(s) by driver
        if 'Driver' in df.columns:
            # single fastest row
            fastest_idx = df['pred_Race_Time'].idxmin()
            fastest = df.loc[fastest_idx]
            st.write(f"**Fastest predicted driver:** {fastest['Driver']} — {fastest['pred_Race_Time']:.3f}")

            actual_idx = df['Race_Time'].idxmin()
            actual = df.loc[actual_idx]
            st.write(f"**Fastest actual driver:** {actual['Driver']} — {actual['Race_Time']:.3f}")

            if fastest['Driver'] == actual['Driver']:
                st.success(f"Prediction matched the real fastest driver ({actual['Driver']}).")
            else:
                delta = fastest['pred_Race_Time'] - actual['Race_Time']
                direction = "slower" if delta > 0 else "faster"
                st.info(f"Predicted fastest ({fastest['Driver']}) is {abs(delta):.3f} seconds {direction} than real fastest ({actual['Driver']}).")
            # top N drivers by their best predicted lap
            top_n = 10
            best_by_driver = (
                df.groupby('Driver', as_index=False)['pred_Race_Time']
                .min()
                .sort_values('pred_Race_Time')
                .head(top_n)
            )
            st.write(f"**Top {top_n} drivers (best predicted lap):**")
            st.table(best_by_driver)

            # Plot top N as bar chart
            fig_top = px.bar(best_by_driver, x='pred_Race_Time', y='Driver', orientation='h',
                            labels={'pred_Race_Time': 'Best Predicted Race Time', 'Driver': 'Driver'},
                            title=f'Top {top_n} Drivers by Best Predicted Lap')
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.warning("Uploaded data has no `Driver` column — include driver names to report fastest driver.")




        