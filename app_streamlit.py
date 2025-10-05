# app_streamlit.py
import streamlit as st
import pandas as pd
import joblib
from urllib.parse import urlparse
import ipaddress
import io

st.set_page_config(page_title="كشف الروابط الضارة", layout="centered")

st.title("🚨 كاشف الروابط الضارة (AI)")

########################
# دوال استخراج الخصائص
########################
def extract_features_single_advanced(url: str):
    url = str(url).strip()
    features = {}
    features["url_length"] = len(url)
    features["count_dots"] = url.count('.')
    features["count_hyphens"] = url.count('-')
    features["count_digits"] = sum(c.isdigit() for c in url)
    features["has_https"] = 1 if url.lower().startswith("https://") else 0
    try:
        features["domain_length"] = len(urlparse(url).netloc)
    except:
        features["domain_length"] = 0
    features["count_slashes"] = url.count('/')
    try:
        netloc = urlparse(url).netloc
        # netloc may include port -> remove port
        netloc_no_port = netloc.split(':')[0]
        ipaddress.ip_address(netloc_no_port)
        features["has_ip"] = 1
    except:
        features["has_ip"] = 0
    suspicious_words = ["login","secure","bank","update","verify","confirm","account","password",
                        "paypal","reset","signin","admin","auth"]
    features["suspicious_words"] = sum(word in url.lower() for word in suspicious_words)
    return pd.DataFrame([features])


def extract_features_batch(df_urls: pd.Series):
    rows = []
    for u in df_urls.astype(str):
        rows.append(extract_features_single_advanced(u).iloc[0].to_dict())
    feat_df = pd.DataFrame(rows)
    return feat_df

########################
# تحميل النموذج (من الملف أو رفع)
########################
@st.cache_resource
def load_model_from_path(path="xgb_model_advanced.pkl"):
    try:
        model = joblib.load(path)
        return model, f"Loaded model from {path}"
    except Exception as e:
        return None, f"Cannot load model from {path}: {e}"

st.sidebar.header("إعدادات النموذج")
model_path_input = st.sidebar.text_input("مسار ملف النموذج (افتراضي)", "xgb_model_advanced.pkl")

model, msg = load_model_from_path(model_path_input)
st.sidebar.write(msg)

# uploader كخيار بديل لو الملف مش موجود
uploaded_model = st.sidebar.file_uploader("أو ارفع ملف نموذج (.pkl)", type=["pkl","joblib"])
if uploaded_model is not None:
    try:
        # joblib.load يعمل مع file-like في معظم الحالات؛ لو فشل نجرب pickle
        uploaded_model.seek(0)
        model = joblib.load(uploaded_model)
        st.sidebar.success("تم تحميل النموذج من الملف المرفوع")
    except Exception as e:
        st.sidebar.error(f"فشل تحميل الملف المرفوع: {e}")
        model = None

if model is None:
    st.warning("لا يوجد نموذج محمّل. إما ضع ملف xgb_model_advanced.pkl في المجلد أو ارفعه من الشريط الجانبي.")
else:
    st.success("النموذج جاهز للاستخدام ✅")

########################
# واجهة اختبار رابط واحد
########################
st.header("اختبار رابط واحد")
single_url = st.text_input("أدخل الرابط لاختباره", placeholder="https://example.com/..")

col1, col2 = st.columns([1,1])
with col1:
    if st.button("تحقق الآن"):
        if not model:
            st.error("لا يوجد نموذج جاهز. حمّله أولاً.")
        elif not single_url:
            st.warning("أدخل رابطًا للاختبار.")
        else:
            feats = extract_features_single_advanced(single_url).to_numpy()
            try:
                pred = model.predict(feats)[0]
                label = "⚠️ ضار" if int(pred)==1 else "✅ غير ضار"
                st.info(f"النتيجة: {label}")
            except Exception as e:
                st.error(f"حدث خطأ أثناء التنبؤ: {e}")

with col2:
    if st.button("تفصيل الخصائص"):
        if not single_url:
            st.warning("أدخل رابطًا أولاً.")
        else:
            st.write(extract_features_single_advanced(single_url).T)

########################
# واجهة فحص ملف CSV (دفعي)
########################
st.header("فحص ملف CSV (قائمة روابط)")
st.write("رفع CSV يحتوي عمود واحد اسمه url أو عمود يحتوي روابط. ستُعاد النتيجة مع عمود prediction (0=آمن,1=ضار).")
uploaded_csv = st.file_uploader("ارفع ملف CSV للبحث (يجب أن يحتوي على عمود url أو link)", type=["csv"])
if uploaded_csv is not None:
    try:
        df_in = pd.read_csv(uploaded_csv)
        # حاول إيجاد عمود URL
        url_col = None
        for c in df_in.columns:
            if c.lower() in ("url","link","uri","website"):
                url_col = c
                break
        if url_col is None:
            # خمن العمود اللي يحتوي روابط بفحص العينات
            candidate = None
            for c in df_in.columns:
                sample = df_in[c].astype(str).head(50).str.lower()
                if sample.str.contains("http://|https://|www\\.|\\.com|\\.net").any():
                    candidate = c
                    break
            url_col = candidate

        if url_col is None:
            st.error("لم أجد عمود روابط في CSV. تأكد أن الملف يحتوي عمودًا يحتوي على روابط.")
        else:
            st.write(f"استخدمت العمود: {url_col}")
            urls_series = df_in[url_col].astype(str)
            feat_df = extract_features_batch(urls_series)
            if model is None:
                st.error("لا يوجد نموذج لتحليل البيانات.")
            else:
                preds = model.predict(feat_df.to_numpy())
                df_out = df_in.copy()
                df_out["prediction"] = preds.astype(int)
                st.success("تم التنبؤ! يمكنك مراجعة النتائج أدناه.")
                st.dataframe(df_out.head(200))
                # تنزيل النتائج
                towrite = io.BytesIO()
                df_out.to_csv(towrite, index=False)
                towrite.seek(0)
                st.download_button("⬇️ تحميل النتائج كـ CSV", towrite, file_name="urls_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"خطأ قراءة الملف: {e}")

########################
# شروحات وملاحظات
########################
st.markdown("---")
st.markdown("### ملاحظات مهمة")
st.markdown("""
- هذا النموذج لا يزور الروابط ولا ينفّذها — فقط يحلل نص الرابط (مفيد للحماية والخصوصية).  
- لو النتيجة خاطئة لبعض الروابط، جرّب تحسين مجموعة البيانات أو إضافة خصائص جديدة.  
- لتشغيل هذا التطبيق على الإنترنت، انشر المجلد على GitHub واربطه بـ Streamlit Cloud (شرح بالأسفل).  
""")