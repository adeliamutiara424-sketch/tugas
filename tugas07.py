# ============================================================
# TOKOKITA - SEGMENTASI PELANGGAN (RFM + K-MEANS)
# Sumber data : HANYA sheet "Data Transaksi Mentah"
# Alur        : Data Cleaning -> RFM -> Preprocessing -> Elbow
#               -> K-Means -> (K Manual) -> Segmentasi -> Dashboard BI
# Style       : Teal / Slate / Mist (kalem, modern)
# ============================================================

import os
import re
import io
import json
import traceback
from functools import wraps

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template_string, request, redirect, url_for,
    flash, send_from_directory, session, send_file
)
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)
app.secret_key = "tokokita_segmentasi_secret_2026"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

USERNAME_VALID = "adel"
PASSWORD_VALID = "adelia123"
REFERENCE_DATE = pd.Timestamp('2024-06-30')
RAW_SHEET_HINTS = ['mentah', 'matah', 'raw']          # nama sheet yang dipakai
NA_TOKENS = {'', 'nan', 'none', 'null', 'na', 'n/a', '-', '--', '#n/a', 'nat'}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ============================================================
# GLOBAL STATE
# ============================================================
UPLOADED_DATA = {}      # hanya berisi sheet 'Data Transaksi Mentah'
CLEANING_DATA = {}
RFM_DATA = {}
PREP_DATA = {}
ELBOW_DATA = {}
KMEANS_DATA = {}
KMANUAL_DATA = {}
DASHBOARD_DATA = {}


# ============================================================
# HELPER
# ============================================================
def format_rupiah(value):
    try:
        if value is None or pd.isna(value):
            return 'Rp 0'
        return f"Rp {int(round(float(value))):,}".replace(',', '.')
    except Exception:
        return str(value)


def format_int(value):
    try:
        return f"{int(round(float(value))):,}".replace(',', '.')
    except Exception:
        return str(value)


def normalize_colname(s):
    return re.sub(r"[\s_\-]+", "", str(s).lower().strip())


def to_json_records(df):
    """DataFrame -> list of dict yang aman untuk JSON (NaN -> null)."""
    return json.loads(df.to_json(orient='records', date_format='iso'))


def fmt_cell(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    if isinstance(v, pd.Timestamp):
        return v.strftime('%Y-%m-%d')
    if isinstance(v, (int, np.integer)):
        return format_int(v)
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer() and abs(v) >= 1000:
            return format_int(v)
        return f"{float(v):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return str(v)


def df_to_table(df, n=50):
    """Kembalikan (headers, rows[str]) untuk ditampilkan di template."""
    sub = df.head(n)
    headers = [str(c) for c in sub.columns]
    rows = [[fmt_cell(v) for v in r] for r in sub.itertuples(index=False, name=None)]
    return headers, rows


# ============================================================
# AUTO MAP KOLOM
# ============================================================
COLUMN_ALIASES = {
    "TransactionID": ["transactionid", "trxid", "invoice", "invoiceno", "notransaksi",
                      "idtransaksi", "transaction", "orderid", "id"],
    "CustomerID":    ["customerid", "idcustomer", "idpelanggan", "memberid", "custid", "pelanggan"],
    "CustomerName":  ["customername", "namapelanggan", "nama", "name"],
    "ProductID":     ["productid", "productcode", "kodeproduk", "sku", "itemcode", "itemid", "idproduk"],
    "ProductName":   ["productname", "namaproduk", "product", "item", "itemname", "deskripsi", "description"],
    "Qty":           ["qty", "quantity", "jumlah", "jumlahbeli", "kuantitas", "unitssold", "terjual"],
    "UnitPrice":     ["unitprice", "harga", "hargasatuan", "price", "unitcost", "hargaproduk"],
    "DiscountPct":   ["discountpct", "diskon", "discount", "discpct", "diskonpct", "discountpercent"],
    "TotalPrice":    ["totalprice", "total", "totalharga", "totalbayar", "amount", "subtotal",
                      "grandtotal", "revenue", "pendapatan", "netsales", "totalsales"],
    "Date":          ["date", "tanggal", "transactiondate", "trxdate", "tgl", "tanggaltransaksi", "tgltransaksi"],
    "Category":      ["category", "kategori", "productcategory", "kategoriproduk"],
    "City":          ["city", "kota", "kabupaten"],
    "Gender":        ["gender", "jeniskelamin", "jk"],
}


def auto_map_columns(df):
    mapping, used = {}, set()
    for std_col, aliases in COLUMN_ALIASES.items():
        for actual in df.columns:
            if actual in used:
                continue
            if normalize_colname(actual) in aliases:
                mapping[actual] = std_col
                used.add(actual)
                break
    return df.rename(columns=mapping) if mapping else df


def get_raw_sheet():
    """Hanya sheet 'Data Transaksi Mentah' yang dipakai."""
    for name, df in UPLOADED_DATA.items():
        return name, df.copy()
    return None, None


def find_raw_sheet_name(sheet_names):
    for n in sheet_names:
        low = n.lower()
        if any(h in low for h in RAW_SHEET_HINTS):
            return n
    return None


# ============================================================
# PARSER
# ============================================================
def is_na(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if v is pd.NaT:
        return True
    return str(v).strip().lower() in NA_TOKENS


def parse_number(v):
    if is_na(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().lower().replace('rp', '').replace(' ', '').replace('%', '')
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        parts = s.split(',')
        s = s.replace(',', '') if (len(parts) > 2 or len(parts[-1]) == 3) else s.replace(',', '.')
    elif s.count('.') > 1 or re.fullmatch(r'-?\d{1,3}(\.\d{3})+', s):
        s = s.replace('.', '')
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_dates(series):
    s = series.copy()
    s = s.where(~s.map(is_na), None)
    try:
        return pd.to_datetime(s, errors='coerce', dayfirst=True, format='mixed')
    except Exception:
        return pd.to_datetime(s, errors='coerce', dayfirst=True)


def clean_text(v):
    if is_na(v):
        return np.nan
    return re.sub(r"\s+", " ", str(v)).strip()


# ============================================================
# BAGIAN 0 - DATA CLEANING
# ============================================================
def run_data_cleaning():
    global CLEANING_DATA
    res = {'success': False, 'error': None, 'sheet_name': None}
    try:
        sheet_name, df = get_raw_sheet()
        if df is None:
            res['error'] = "Sheet 'Data Transaksi Mentah' belum diunggah."
            CLEANING_DATA = res
            return res
        res['sheet_name'] = sheet_name
        df = auto_map_columns(df)
        df.columns = df.columns.astype(str).str.strip()

        required = ['CustomerID', 'Date', 'Qty', 'UnitPrice']
        miss_cols = [c for c in required if c not in df.columns]
        if miss_cols:
            res['error'] = f"Kolom wajib tidak ditemukan: {', '.join(miss_cols)}. Kolom terbaca: {', '.join(df.columns)}"
            CLEANING_DATA = res
            return res

        # ---------- Tugas 0.1 : identifikasi masalah (data mentah) ----------
        raw = df.copy()
        rows_before, cols_before = len(raw), len(raw.columns)
        missing_by_col = {}
        for c in raw.columns:
            n = int(raw[c].map(is_na).sum())
            if n:
                missing_by_col[c] = n
        missing_before = sum(missing_by_col.values())

        raw_txt = raw.astype(str).apply(lambda col: col.str.strip())
        dup_exact_before = int(raw_txt.duplicated().sum())
        dup_examples = raw[raw_txt.duplicated(keep=False)].head(10)

        qty_raw = raw['Qty'].map(parse_number)
        price_raw = raw['UnitPrice'].map(parse_number)
        date_raw = parse_dates(raw['Date'])
        inv_qty = int((qty_raw <= 0).sum())
        inv_price = int((price_raw <= 0).sum())
        inv_date = int(date_raw.isna().sum() - raw['Date'].map(is_na).sum()) + int((date_raw > REFERENCE_DATE).sum())
        inv_cust = int(raw['CustomerID'].map(
            lambda v: (not is_na(v)) and bool(re.search(r'guest|unknown|test|dummy|xxx', str(v).lower()))).sum())

        # ---------- Tugas 0.2 : hapus duplikat persis ----------
        df = df.loc[~raw_txt.duplicated()].reset_index(drop=True)

        # ---------- Tugas 0.4 : standarisasi ----------
        log = []
        df['CustomerID'] = df['CustomerID'].map(clean_text).str.upper()
        log.append(('CustomerID', 'Trim, hapus spasi ganda, UPPERCASE', 'ID pelanggan seragam sehingga satu pelanggan = satu ID'))
        if 'TransactionID' in df.columns:
            df['TransactionID'] = df['TransactionID'].map(clean_text).str.upper()
            log.append(('TransactionID', 'Trim & UPPERCASE', 'ID transaksi seragam'))
        if 'ProductID' in df.columns:
            df['ProductID'] = df['ProductID'].map(clean_text).str.upper()
        if 'CustomerName' in df.columns:
            df['CustomerName'] = df['CustomerName'].map(clean_text).str.title()
            log.append(('CustomerName', 'Trim & Title Case', 'Penulisan nama pelanggan diseragamkan'))
        for c in ['ProductName', 'Category', 'City', 'Gender']:
            if c in df.columns:
                df[c] = df[c].map(clean_text).str.title()
                log.append((c, 'Trim & Title Case', 'Variasi huruf besar/kecil & spasi diseragamkan'))
        df['Date'] = parse_dates(df['Date'])
        log.append(('Date', 'Konversi ke datetime (YYYY-MM-DD)', 'Berbagai format tanggal disatukan'))
        for c in ['Qty', 'UnitPrice', 'DiscountPct', 'TotalPrice']:
            if c in df.columns:
                df[c] = df[c].map(parse_number)
                log.append((c, 'Konversi ke numerik', "Simbol 'Rp', titik/koma ribuan dibersihkan"))
        if 'DiscountPct' in df.columns and df['DiscountPct'].max() > 1:
            df['DiscountPct'] = df['DiscountPct'] / 100.0
            log.append(('DiscountPct', 'Persen -> pecahan', 'Nilai 10 menjadi 0,10 agar rumus total konsisten'))
        res['standardization_log'] = [{'kolom': a, 'aksi': b, 'detail': c} for a, b, c in log]

        # ---------- Tugas 0.3 : missing value ----------
        mrep = []
        def add(col, n, method, reason):
            mrep.append({'kolom': col, 'n': int(n), 'metode': method, 'alasan': reason})

        if 'CustomerName' in df.columns:
            n = df['CustomerName'].isna().sum()
            df['CustomerName'] = df['CustomerName'].fillna('Tidak Diketahui')
            add('CustomerName', n, "Isi 'Tidak Diketahui'", 'Nama tidak dipakai di RFM; baris transaksi tetap valid.')
        for c in ['ProductName', 'Category', 'City', 'Gender']:
            if c in df.columns:
                n = df[c].isna().sum()
                df[c] = df[c].fillna('Tidak Diketahui')
                add(c, n, "Isi 'Tidak Diketahui'", 'Atribut deskriptif, tidak memengaruhi R/F/M.')
        if 'DiscountPct' in df.columns:
            n = df['DiscountPct'].isna().sum()
            df['DiscountPct'] = df['DiscountPct'].fillna(0).clip(0, 1)
            add('DiscountPct', n, 'Isi 0', 'Diskon kosong dianggap tidak ada diskon.')
        else:
            df['DiscountPct'] = 0.0
        n_cid = int(df['CustomerID'].isna().sum())
        n_date = int(df['Date'].isna().sum())
        n_qty = int(df['Qty'].isna().sum())
        n_prc = int(df['UnitPrice'].isna().sum())
        add('CustomerID', n_cid, 'Hapus baris', 'Transaksi tanpa pelanggan tidak bisa dihitung per pelanggan.')
        add('Date', n_date, 'Hapus baris', 'Tanpa tanggal, Recency tidak dapat dihitung.')
        add('Qty', n_qty, 'Hapus baris', 'Jumlah tidak boleh ditebak karena memengaruhi Monetary.')
        add('UnitPrice', n_prc, 'Hapus baris', 'Harga tidak boleh ditebak karena memengaruhi Monetary.')
        n_tot = int(df['TotalPrice'].isna().sum()) if 'TotalPrice' in df.columns else len(df)
        add('TotalPrice', n_tot, 'Hitung ulang Qty x UnitPrice x (1 - Diskon)', 'Total dihitung dari komponen yang valid.')
        df = df.dropna(subset=['CustomerID', 'Date', 'Qty', 'UnitPrice'])
        res['missing_report'] = mrep

        # ---------- Tugas 0.5 : nilai tidak valid ----------
        n0 = len(df)
        invalid_log = []
        m = df['Qty'] <= 0
        invalid_log.append(('Qty <= 0', int(m.sum())));        df = df[~m]
        m = df['UnitPrice'] <= 0
        invalid_log.append(('UnitPrice <= 0', int(m.sum())));  df = df[~m]
        m = df['Date'] > REFERENCE_DATE
        invalid_log.append((f"Tanggal setelah {REFERENCE_DATE.strftime('%d %b %Y')}", int(m.sum())));  df = df[~m]
        m = df['CustomerID'].str.contains(r'GUEST|UNKNOWN|TEST|DUMMY|XXX', regex=True, na=False) | (df['CustomerID'].str.len() < 3)
        invalid_log.append(('ID pelanggan tidak terdaftar / tidak valid', int(m.sum())));  df = df[~m]

        # Duplikat transaksi: ID beda, isi sama
        key = [c for c in ['CustomerID', 'Date', 'ProductID', 'Qty', 'UnitPrice'] if c in df.columns]
        dup_mask = df.duplicated(subset=key, keep='first')
        n_dup_trx = int(dup_mask.sum())
        dup_trx_examples = df[df.duplicated(subset=key, keep=False)].sort_values(key).head(10)
        df = df[~dup_mask]

        # Total tidak konsisten
        expected = (df['Qty'] * df['UnitPrice'] * (1 - df['DiscountPct'])).round(2)
        if 'TotalPrice' in df.columns:
            diff = (df['TotalPrice'] - expected).abs()
            inconsistent = diff.isna() | (diff > (expected.abs() * 0.01 + 1))
            n_incons = int((inconsistent & df['TotalPrice'].notna()).sum())
        else:
            n_incons = 0
        df['TotalPrice'] = expected
        invalid_log.append(('Total tidak sama dengan Qty x Harga x (1-Diskon) -> dihitung ulang', n_incons))

        df = df.sort_values('Date').reset_index(drop=True)

        # ---------- Tugas 0.6 : data bersih ----------
        rows_after = len(df)
        res['quality'] = [
            {'metrik': 'Jumlah baris', 'sebelum': rows_before, 'sesudah': rows_after},
            {'metrik': 'Jumlah kolom', 'sebelum': cols_before, 'sesudah': len(df.columns)},
            {'metrik': 'Nilai kosong', 'sebelum': missing_before, 'sesudah': int(df.isna().sum().sum())},
            {'metrik': 'Duplikat persis', 'sebelum': dup_exact_before, 'sesudah': int(df.duplicated().sum())},
            {'metrik': 'Duplikat transaksi (ID beda, isi sama)', 'sebelum': n_dup_trx, 'sesudah': int(df.duplicated(subset=key).sum())},
            {'metrik': 'Qty <= 0', 'sebelum': inv_qty, 'sesudah': int((df['Qty'] <= 0).sum())},
            {'metrik': 'UnitPrice <= 0', 'sebelum': inv_price, 'sesudah': int((df['UnitPrice'] <= 0).sum())},
            {'metrik': 'Tanggal tidak valid / melewati tanggal acuan', 'sebelum': inv_date, 'sesudah': int((df['Date'] > REFERENCE_DATE).sum())},
            {'metrik': 'ID pelanggan tidak valid', 'sebelum': inv_cust, 'sesudah': 0},
            {'metrik': 'Total tidak konsisten', 'sebelum': n_incons, 'sesudah': 0},
        ]
        res['missing_by_col'] = missing_by_col
        res['invalid_log'] = [{'jenis': a, 'n': b} for a, b in invalid_log]
        res['dup_examples'] = df_to_table(dup_examples, 5) if len(dup_examples) else None
        res['dup_trx_examples'] = df_to_table(dup_trx_examples, 6) if len(dup_trx_examples) else None
        res['n_dup_exact'] = dup_exact_before
        res['n_dup_trx'] = n_dup_trx
        res['n_missing_before'] = missing_before
        res['rows_before'] = rows_before
        res['rows_after'] = rows_after
        res['cols_before'] = cols_before
        res['clean_df'] = df
        res['clean_table'] = df_to_table(df, 15)
        res['success'] = True
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    CLEANING_DATA = res
    return res


# ============================================================
# BAGIAN 1 - RFM
# ============================================================
def run_rfm():
    global RFM_DATA
    res = {'success': False, 'error': None}
    try:
        if not CLEANING_DATA.get('success'):
            res['error'] = 'Lakukan Data Cleaning terlebih dahulu.'
            RFM_DATA = res
            return res
        df = CLEANING_DATA['clean_df']
        id_col = 'TransactionID' if 'TransactionID' in df.columns else 'Date'
        g = df.groupby('CustomerID')
        rfm = pd.DataFrame({
            'Recency': (REFERENCE_DATE - g['Date'].max()).dt.days,
            'Frequency': g[id_col].nunique(),
            'Monetary': g['TotalPrice'].sum().round(2),
        }).reset_index()
        rfm = rfm.sort_values('Monetary', ascending=False).reset_index(drop=True)
        res.update({
            'rfm_df': rfm,
            'table': df_to_table(rfm, 50),
            'n_customers': len(rfm),
            'avg_recency': float(rfm['Recency'].mean()),
            'avg_frequency': float(rfm['Frequency'].mean()),
            'avg_monetary': float(rfm['Monetary'].mean()),
            'success': True,
        })
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    RFM_DATA = res
    return res


RFM_TEMPLATE_COLS = ['ID_Pelanggan', 'Nama_Pelanggan', 'Last_Transaction', 'Recency', 'Frequency', 'Monetary',
                     'Log_Frequency', 'Log_Monetary', 'Z_Recency', 'Z_Log_Frequency', 'Z_Log_Monetary',
                     'Cluster', 'Segmen_Bisnis']


def _full_pipeline_frames():
    """Hitung Log, Z-score, dan Cluster/Segmen untuk ditampilkan di tabel RFM.
    Memakai fungsi yang sama dengan menu Transformasi, Elbow, dan K-Means, tetapi state menu
    tersebut dikembalikan seperti semula (sidebar tidak otomatis bertanda selesai)."""
    global PREP_DATA, ELBOW_DATA, KMEANS_DATA
    saved = (PREP_DATA, ELBOW_DATA, KMEANS_DATA)
    prep_df, lab_df = None, None
    try:
        if not PREP_DATA.get('success'):
            run_preprocessing()
        if PREP_DATA.get('success'):
            if not ELBOW_DATA.get('success'):
                run_elbow()
            if ELBOW_DATA.get('success') and not KMEANS_DATA.get('success'):
                run_kmeans(ELBOW_DATA.get('optimal_k', 3))
            prep_df = PREP_DATA.get('prep_df')
            lab_df = KMEANS_DATA.get('labeled_df') if KMEANS_DATA.get('success') else None
    finally:
        PREP_DATA, ELBOW_DATA, KMEANS_DATA = saved
    return prep_df, lab_df


def build_rfm_template():
    """Susun tabel RFM sesuai template lembar kerja.
    Seluruh kolom (Log, Z, Cluster, Segmen_Bisnis) dihitung otomatis dari data bersih."""
    if not RFM_DATA.get('success'):
        return None
    clean = CLEANING_DATA['clean_df']
    df = RFM_DATA['rfm_df'][['CustomerID', 'Recency', 'Frequency', 'Monetary']].copy()

    info = clean.groupby('CustomerID').agg(Last_Transaction=('Date', 'max'))
    if 'CustomerName' in clean.columns:
        def pick_name(s):
            m = s.mode()
            return m.iloc[0] if len(m) else s.iloc[0]
        info['Nama_Pelanggan'] = clean.groupby('CustomerID')['CustomerName'].agg(pick_name)
    else:
        info['Nama_Pelanggan'] = '-'
    df = df.merge(info.reset_index(), on='CustomerID', how='left')

    prep_df, lab_df = _full_pipeline_frames()
    if prep_df is not None:
        p = prep_df[['CustomerID', 'Log_Frequency', 'Log_Monetary', 'Scaled_Recency',
                                  'Scaled_Log_Frequency', 'Scaled_Log_Monetary']].rename(columns={
            'Scaled_Recency': 'Z_Recency', 'Scaled_Log_Frequency': 'Z_Log_Frequency',
            'Scaled_Log_Monetary': 'Z_Log_Monetary'})
        df = df.merge(p, on='CustomerID', how='left')
    else:
        for c in ['Log_Frequency', 'Log_Monetary', 'Z_Recency', 'Z_Log_Frequency', 'Z_Log_Monetary']:
            df[c] = np.nan

    if lab_df is not None:
        k = lab_df[['CustomerID', 'Cluster', 'Label_Bisnis']].rename(
            columns={'Label_Bisnis': 'Segmen_Bisnis'})
        df = df.merge(k, on='CustomerID', how='left')
    else:
        df['Cluster'] = np.nan
        df['Segmen_Bisnis'] = np.nan

    df = df.rename(columns={'CustomerID': 'ID_Pelanggan'})
    return df[RFM_TEMPLATE_COLS].sort_values('ID_Pelanggan').reset_index(drop=True)


def rfm_template_display(df, n=50):
    """Format tabel template untuk tampilan (kolom yang belum dihitung tampil '-')."""
    d = df.head(n).copy()
    d['Last_Transaction'] = pd.to_datetime(d['Last_Transaction']).dt.strftime('%Y-%m-%d')
    d['Monetary'] = d['Monetary'].map(lambda v: format_int(v))
    for c in ['Log_Frequency', 'Log_Monetary', 'Z_Recency', 'Z_Log_Frequency', 'Z_Log_Monetary']:
        d[c] = d[c].map(lambda v: '-' if pd.isna(v) else f"{v:.4f}".replace('.', ','))
    d['Cluster'] = d['Cluster'].map(lambda v: '-' if pd.isna(v) else str(int(v)))
    d['Segmen_Bisnis'] = d['Segmen_Bisnis'].map(lambda v: '-' if pd.isna(v) else str(v))
    return df_to_table(d, n)


# ============================================================
# BAGIAN 2 - LOG TRANSFORM & STANDARDISASI
# ============================================================
def run_preprocessing():
    global PREP_DATA
    res = {'success': False, 'error': None}
    try:
        if not RFM_DATA.get('success'):
            res['error'] = 'Lakukan RFM terlebih dahulu.'
            PREP_DATA = res
            return res
        rfm = RFM_DATA['rfm_df'].copy()
        rfm['Log_Frequency'] = np.log1p(rfm['Frequency'])
        rfm['Log_Monetary'] = np.log1p(rfm['Monetary'])
        feats = ['Recency', 'Log_Frequency', 'Log_Monetary']
        scaled = StandardScaler().fit_transform(rfm[feats])
        rfm['Scaled_Recency'] = scaled[:, 0]
        rfm['Scaled_Log_Frequency'] = scaled[:, 1]
        rfm['Scaled_Log_Monetary'] = scaled[:, 2]

        stats = []
        for lbl, col in [('Recency', 'Scaled_Recency'), ('Log_Frequency', 'Scaled_Log_Frequency'),
                         ('Log_Monetary', 'Scaled_Log_Monetary')]:
            stats.append({'fitur': lbl, 'mean': float(rfm[col].mean()), 'std': float(rfm[col].std(ddof=0))})
        skew = {
            'Frequency': float(RFM_DATA['rfm_df']['Frequency'].skew()),
            'Log_Frequency': float(rfm['Log_Frequency'].skew()),
            'Monetary': float(RFM_DATA['rfm_df']['Monetary'].skew()),
            'Log_Monetary': float(rfm['Log_Monetary'].skew()),
        }
        res.update({'prep_df': rfm, 'table': df_to_table(rfm, 50), 'stats': stats, 'skew': skew, 'success': True})
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    PREP_DATA = res
    return res


SCALED_COLS = ['Scaled_Recency', 'Scaled_Log_Frequency', 'Scaled_Log_Monetary']


# ============================================================
# BAGIAN 3 - ELBOW
# ============================================================
def detect_elbow(ks, inertia):
    """Titik dengan jarak terjauh dari garis lurus k pertama -> k terakhir."""
    x = np.array(ks, dtype=float)
    y = np.array(inertia, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min() if y.max() != y.min() else 1)
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    d = [abs(np.cross(p2 - p1, p1 - np.array([a, b]))) / np.linalg.norm(p2 - p1) for a, b in zip(xn, yn)]
    return int(ks[int(np.argmax(d))])


def run_elbow():
    global ELBOW_DATA
    res = {'success': False, 'error': None, 'k_values': [2, 3, 4, 5, 6]}
    try:
        if not PREP_DATA.get('success'):
            res['error'] = 'Lakukan Preprocessing terlebih dahulu.'
            ELBOW_DATA = res
            return res
        X = PREP_DATA['prep_df'][SCALED_COLS].values
        inertia = []
        for k in res['k_values']:
            inertia.append(float(KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500).fit(X).inertia_))
        drops = [(inertia[i] - inertia[i + 1]) / inertia[i] * 100 for i in range(len(inertia) - 1)]
        res.update({'inertia': inertia, 'drops': drops, 'optimal_k': detect_elbow(res['k_values'], inertia), 'success': True})
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    ELBOW_DATA = res
    return res


# ============================================================
# BAGIAN 4 - K-MEANS & INTERPRETASI
# ============================================================
def run_kmeans(k):
    global KMEANS_DATA
    res = {'success': False, 'error': None, 'k': k}
    try:
        if not ELBOW_DATA.get('success'):
            res['error'] = 'Lakukan Elbow Method terlebih dahulu.'
            KMEANS_DATA = res
            return res
        rfm = PREP_DATA['prep_df'].copy()
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500).fit(rfm[SCALED_COLS].values)
        rfm['Cluster'] = km.labels_

        prof = rfm.groupby('Cluster').agg(
            Jumlah_Pelanggan=('CustomerID', 'count'),
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean'),
            Avg_Log_Frequency=('Log_Frequency', 'mean'),
            Avg_Log_Monetary=('Log_Monetary', 'mean'),
        ).reset_index()

        # skor nilai pelanggan: recency rendah, frequency & monetary tinggi
        z = lambda s: (s - s.mean()) / (s.std(ddof=0) or 1)
        prof['Skor'] = -z(prof['Avg_Recency']) + z(prof['Avg_Log_Frequency']) + z(prof['Avg_Log_Monetary'])
        order = prof.sort_values('Skor', ascending=False)['Cluster'].tolist()
        mean_rec = rfm['Recency'].mean()
        labels = {order[0]: 'Loyal'}
        for c in order[1:]:
            r = float(prof.loc[prof.Cluster == c, 'Avg_Recency'].iloc[0])
            labels[c] = 'New' if (r < mean_rec and 'New' not in labels.values()) else 'Churn Risk'
        # pastikan ada Churn Risk bila k >= 2 dan semua sudah New/Loyal
        if k >= 2 and 'Churn Risk' not in labels.values():
            worst = order[-1]
            labels[worst] = 'Churn Risk'
        prof['Label_Bisnis'] = prof['Cluster'].map(labels)

        kar, strat = [], []
        for _, r in prof.iterrows():
            ring = (f"Rata-rata Recency {r.Avg_Recency:.0f} hari, Frequency {r.Avg_Frequency:.1f}x, "
                    f"Monetary {format_rupiah(r.Avg_Monetary)}.")
            if r.Label_Bisnis == 'Loyal':
                kar.append(ring + ' Belanja baru-baru ini, sering, dan bernilai besar.')
                strat.append('Program loyalitas (poin & tier), akses awal produk baru, reward referral, bundling premium. Hindari diskon besar.')
            elif r.Label_Bisnis == 'New':
                kar.append(ring + ' Aktif baru-baru ini namun frekuensi dan nilai belanja masih rendah.')
                strat.append('Welcome voucher, onboarding & edukasi produk, rekomendasi produk pelengkap, follow-up untuk repeat order ke-2 dan ke-3.')
            else:
                kar.append(ring + ' Lama tidak bertransaksi dengan frekuensi/nilai belanja relatif rendah.')
                strat.append('Kampanye win-back berbatas waktu, survei alasan berhenti, pengingat personal via WhatsApp/email, hentikan iklan massal ke segmen ini.')
        prof['Karakteristik'] = kar
        prof['Strategi'] = strat

        rfm = rfm.merge(prof[['Cluster', 'Label_Bisnis']], on='Cluster', how='left')

        comp = rfm.groupby('Label_Bisnis').agg(
            Jumlah=('CustomerID', 'count'), Revenue=('Monetary', 'sum'),
            Avg_Monetary=('Monetary', 'mean'), Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean')).reset_index()

        res.update({
            'labeled_df': rfm,
            'profile': prof.drop(columns=['Skor']),
            'composition': comp,
            'iterations': int(km.n_iter_),
            'table': df_to_table(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'Label_Bisnis']], 50),
            'scatter': to_json_records(rfm[['Frequency', 'Monetary', 'Label_Bisnis']].sample(min(len(rfm), 800), random_state=1)),
            'success': True,
        })
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    KMEANS_DATA = res
    return res


# ============================================================
# BAGIAN 4B - K MANUAL (k ditentukan sendiri)
# Cluster tetap bernomor, ditambah KETERANGAN teks otomatis
# yang diturunkan dari profil RFM masing-masing cluster.
# ============================================================
def _level(z, hi=0.5, lo=-0.5):
    """Ubah z-score rata-rata cluster menjadi tingkat Rendah / Sedang / Tinggi."""
    if z >= hi:
        return 'Tinggi'
    if z <= lo:
        return 'Rendah'
    return 'Sedang'


STRATEGI_MANUAL = {
    'Pelanggan Bernilai Tinggi': {
        'tujuan': 'Mempertahankan loyalitas dan memperbesar nilai seumur hidup pelanggan (retensi).',
        'aksi': ['Program loyalitas bertingkat (poin, tier, reward eksklusif)',
                 'Akses awal produk baru dan penawaran khusus member',
                 'Program referral agar pelanggan mengajak pelanggan baru',
                 'Bundling premium; hindari diskon besar karena mereka sudah rutin membeli'],
        'kanal': 'WhatsApp personal, email eksklusif, aplikasi/member card',
        'kpi': 'Retention rate, frekuensi repeat order, jumlah referral',
        'anggaran': 'Tinggi (fokus retensi, bukan diskon)',
    },
    'Pelanggan Potensial': {
        'tujuan': 'Menaikkan frekuensi dan nilai belanja agar naik menjadi pelanggan bernilai tinggi.',
        'aksi': ['Upsell dan cross-sell produk pelengkap',
                 'Paket bundling dan target belanja berhadiah (misal: belanja X dapat bonus)',
                 'Rekomendasi produk berdasarkan riwayat pembelian',
                 'Undangan program loyalitas sebagai pintu masuk tier berikutnya'],
        'kanal': 'Email/WhatsApp tersegmentasi, rekomendasi di aplikasi atau situs',
        'kpi': 'Kenaikan rata-rata nilai transaksi, kenaikan frekuensi, konversi ke segmen bernilai tinggi',
        'anggaran': 'Tinggi (fokus pertumbuhan)',
    },
    'Pelanggan Aktif Baru / Nilai Rendah': {
        'tujuan': 'Mendorong repeat order dan membangun kebiasaan belanja sejak awal.',
        'aksi': ['Welcome voucher untuk pembelian ke-2',
                 'Onboarding dan edukasi produk (panduan, konten singkat)',
                 'Rekomendasi produk pelengkap dari pembelian pertama',
                 'Follow-up terjadwal untuk repeat order ke-2 dan ke-3'],
        'kanal': 'WhatsApp/email otomatis setelah transaksi, notifikasi aplikasi',
        'kpi': 'Repeat purchase rate, waktu menuju pembelian ke-2',
        'anggaran': 'Sedang (terukur per pelanggan baru)',
    },
    'Pelanggan Menengah': {
        'tujuan': 'Menjaga keterlibatan dan mencegah pelanggan bergeser menjadi pasif.',
        'aksi': ['Promo musiman dan penawaran yang relevan dengan kategori favorit',
                 'Rekomendasi produk personal secara berkala',
                 'Pengingat belanja ulang sebelum pelanggan terlalu lama tidak aktif',
                 'Uji A/B pada penawaran untuk melihat pemicu belanja paling efektif'],
        'kanal': 'Email berkala, broadcast WhatsApp, media sosial',
        'kpi': 'Open/click rate, pertumbuhan frekuensi, penurunan jumlah pelanggan yang menjadi pasif',
        'anggaran': 'Sedang',
    },
    'Pelanggan Pasif / Berisiko Hilang': {
        'tujuan': 'Mengaktifkan kembali pelanggan sebelum benar-benar hilang (win-back).',
        'aksi': ['Kampanye win-back dengan penawaran berbatas waktu',
                 'Pengingat personal ("kami kangen") lewat WhatsApp atau email',
                 'Survei singkat alasan berhenti berbelanja',
                 'Hentikan iklan massal ke segmen ini jika tidak merespons setelah 2-3 kali kontak'],
        'kanal': 'WhatsApp/email personal, retargeting terbatas',
        'kpi': 'Reactivation rate, biaya per pelanggan yang kembali, alasan berhenti dari survei',
        'anggaran': 'Selektif (win-back berbatas waktu)',
    },
    'Pelanggan Bernilai Rendah': {
        'tujuan': 'Menjaga hubungan dengan biaya serendah mungkin dan menemukan pelanggan yang layak dinaikkan.',
        'aksi': ['Komunikasi otomatis berbiaya rendah (newsletter, broadcast)',
                 'Hindari promo/diskon mahal pada segmen ini',
                 'Tawarkan paket hemat atau produk entry-level untuk memancing pembelian ulang',
                 'Pantau pelanggan yang menunjukkan kenaikan aktivitas untuk dipindah ke segmen lebih tinggi'],
        'kanal': 'Email massal otomatis, notifikasi aplikasi',
        'kpi': 'Biaya per kontak, persentase pelanggan yang naik segmen',
        'anggaran': 'Rendah (hemat biaya)',
    },
}


def describe_manual_clusters(rfm, prof):
    """Tambahkan keterangan teks pada profil cluster K Manual.

    Dasar penilaian: rata-rata z-score (hasil standardisasi) tiap fitur per cluster.
    z-score >= 0,5 -> Tinggi | z-score <= -0,5 -> Rendah | selain itu -> Sedang.
    Untuk Recency, nilai RENDAH berarti pelanggan baru-baru ini bertransaksi (bagus).
    """
    zmean = rfm.groupby('Cluster')[SCALED_COLS].mean().reset_index()
    zmean.columns = ['Cluster', 'Z_R', 'Z_F', 'Z_M']
    prof = prof.merge(zmean, on='Cluster', how='left')

    rec_txt = {'Rendah': 'baru-baru ini bertransaksi', 'Sedang': 'cukup lama tidak bertransaksi',
               'Tinggi': 'sudah lama tidak bertransaksi'}
    frq_txt = {'Tinggi': 'sering berbelanja', 'Sedang': 'cukup sering berbelanja', 'Rendah': 'jarang berbelanja'}
    mon_txt = {'Tinggi': 'nilai belanja besar', 'Sedang': 'nilai belanja sedang', 'Rendah': 'nilai belanja kecil'}

    names, kars, sarans, lvR, lvF, lvM, skors = [], [], [], [], [], [], []
    tujuans, aksis, kanals, kpis, anggarans = [], [], [], [], []
    for _, r in prof.iterrows():
        lr, lf, lm = _level(r.Z_R), _level(r.Z_F), _level(r.Z_M)
        skor = (-r.Z_R + r.Z_F + r.Z_M) / 3.0
        if skor >= 0.75:
            nama = 'Pelanggan Bernilai Tinggi'
            saran = 'Pertahankan dengan program loyalitas, reward eksklusif, dan akses awal produk baru. Hindari diskon besar.'
        elif lr == 'Tinggi' and skor < 0:
            nama = 'Pelanggan Pasif / Berisiko Hilang'
            saran = 'Kampanye win-back berbatas waktu, pengingat personal (WhatsApp/email), dan survei alasan berhenti.'
        elif lr == 'Rendah' and lf != 'Tinggi' and lm != 'Tinggi':
            nama = 'Pelanggan Aktif Baru / Nilai Rendah'
            saran = 'Welcome voucher, onboarding, rekomendasi produk pelengkap, dan dorongan repeat order.'
        elif skor >= 0.25:
            nama = 'Pelanggan Potensial'
            saran = 'Tawarkan bundling dan upsell untuk menaikkan frekuensi dan nilai belanja menuju segmen bernilai tinggi.'
        elif skor <= -0.5:
            nama = 'Pelanggan Bernilai Rendah'
            saran = 'Gunakan komunikasi hemat biaya (email/broadcast), hindari promosi mahal pada segmen ini.'
        else:
            nama = 'Pelanggan Menengah'
            saran = 'Jaga keterlibatan dengan promo musiman dan rekomendasi produk yang relevan.'
        kar = (f"Rata-rata Recency {r.Avg_Recency:.0f} hari ({lr.lower()}, {rec_txt[lr]}), "
               f"Frequency {r.Avg_Frequency:.1f}x ({lf.lower()}, {frq_txt[lf]}), "
               f"Monetary {format_rupiah(r.Avg_Monetary)} ({lm.lower()}, {mon_txt[lm]}).")
        names.append(nama); kars.append(kar); sarans.append(saran)
        st = STRATEGI_MANUAL[nama]
        aksi = list(st['aksi'])
        extra = []
        if lr == 'Tinggi':
            extra.append('Kirim pengingat personal karena cluster ini sudah lama tidak bertransaksi')
        if lf == 'Rendah':
            extra.append('Dorong repeat order dengan voucher pembelian berikutnya berbatas waktu')
        if lm == 'Rendah':
            extra.append('Tawarkan paket hemat/bundling untuk menaikkan nilai keranjang')
        if lf == 'Tinggi' and lm != 'Tinggi':
            extra.append('Upsell produk bernilai lebih tinggi karena pelanggan sudah sering berbelanja')
        if lm == 'Tinggi' and lf != 'Tinggi':
            extra.append('Dorong frekuensi lewat langganan atau pengingat stok ulang karena nilai belanja sudah besar')
        aksi += [e for e in extra if e not in aksi][:2]
        tujuans.append(st['tujuan']); aksis.append('||'.join(aksi))
        kanals.append(st['kanal']); kpis.append(st['kpi']); anggarans.append(st['anggaran'])
        lvR.append(lr); lvF.append(lf); lvM.append(lm); skors.append(float(skor))

    prof['Keterangan'] = names
    prof['Karakteristik'] = kars
    prof['Saran'] = sarans
    prof['Tujuan'] = tujuans
    prof['Aksi'] = aksis
    prof['Kanal'] = kanals
    prof['KPI'] = kpis
    prof['Anggaran'] = anggarans
    prof['Level_R'] = lvR
    prof['Level_F'] = lvF
    prof['Level_M'] = lvM
    prof['Skor'] = skors
    prof['Peringkat'] = prof['Skor'].rank(ascending=False, method='first').astype(int)

    # Jika beberapa cluster mendapat keterangan yang sama, bedakan dengan ciri RFM
    # yang paling membedakan di antara cluster-cluster kembar tersebut
    # (bukan nomor urut), sehingga setiap cluster punya keterangan unik & jelas.
    # (kolom z, arah urut dari "terbaik", label ujung terbaik, label ujung terburuk)
    features = [
        ('Z_R', True,  'Paling Baru Aktif',        'Paling Lama Tidak Aktif'),
        ('Z_F', False, 'Paling Sering Belanja',    'Paling Jarang Belanja'),
        ('Z_M', False, 'Nilai Belanja Terbesar',   'Nilai Belanja Terkecil'),
    ]
    for nm in list(prof['Keterangan'].unique()):
        idx = prof.index[prof['Keterangan'] == nm].tolist()
        if len(idx) < 2:
            continue
        # pilih fitur dengan selisih terbesar antar cluster kembar
        col, asc, best_lbl, worst_lbl = max(
            features, key=lambda f: prof.loc[idx, f[0]].max() - prof.loc[idx, f[0]].min())
        idx_sorted = sorted(idx, key=lambda i: prof.loc[i, col], reverse=not asc)
        last = len(idx_sorted) - 1
        for pos, i in enumerate(idx_sorted):
            if pos == 0:
                qual = best_lbl
            elif pos == last:
                qual = worst_lbl
            else:
                qual = f"Tingkat Menengah {pos}"
            prof.loc[i, 'Keterangan'] = f"{nm} - {qual}"

    # Pengaman terakhir: bila masih ada yang kembar, tambahkan nomor cluster
    dup = prof['Keterangan'].duplicated(keep=False)
    prof.loc[dup, 'Keterangan'] = prof.loc[dup].apply(
        lambda r: f"{r['Keterangan']} (Cluster {int(r['Cluster'])})", axis=1)
    return prof


def run_kmanual(k):
    """Langsung K-Means dengan k pilihan pengguna dari data hasil transformasi.
    Setiap cluster punya nomor DAN keterangan teks hasil interpretasi profil RFM."""
    global KMANUAL_DATA
    res = {'success': False, 'error': None, 'k': k}
    try:
        if not PREP_DATA.get('success'):
            res['error'] = 'Lakukan Transformasi & Standardisasi terlebih dahulu.'
            KMANUAL_DATA = res
            return res
        rfm = PREP_DATA['prep_df'].copy()
        n = len(rfm)
        if k < 2:
            res['error'] = 'Jumlah cluster (k) minimal 2.'
            KMANUAL_DATA = res
            return res
        if k >= n:
            res['error'] = f'Jumlah cluster (k) harus lebih kecil dari jumlah pelanggan ({n}).'
            KMANUAL_DATA = res
            return res

        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=500).fit(rfm[SCALED_COLS].values)
        rfm['Cluster'] = km.labels_

        prof = rfm.groupby('Cluster').agg(
            Jumlah_Pelanggan=('CustomerID', 'count'),
            Avg_Recency=('Recency', 'mean'),
            Avg_Frequency=('Frequency', 'mean'),
            Avg_Monetary=('Monetary', 'mean'),
            Avg_Log_Frequency=('Log_Frequency', 'mean'),
            Avg_Log_Monetary=('Log_Monetary', 'mean'),
        ).reset_index()

        # keterangan teks untuk setiap cluster
        prof = describe_manual_clusters(rfm, prof)
        rfm = rfm.merge(prof[['Cluster', 'Keterangan']], on='Cluster', how='left')

        res.update({
            'labeled_df': rfm,
            'profile': prof,
            'profile_json': to_json_records(prof),
            'inertia': float(km.inertia_),
            'iterations': int(km.n_iter_),
            'table': df_to_table(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'Keterangan']], 50),
            'scatter': to_json_records(rfm[['Frequency', 'Monetary', 'Cluster']].sample(min(len(rfm), 800), random_state=1)),
            'success': True,
        })
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    KMANUAL_DATA = res
    return res


# ============================================================
# BAGIAN 5 - DASHBOARD BI
# ============================================================
def run_dashboard():
    global DASHBOARD_DATA
    res = {'success': False, 'error': None}
    try:
        if not KMEANS_DATA.get('success'):
            res['error'] = 'Lakukan K-Means terlebih dahulu.'
            DASHBOARD_DATA = res
            return res
        rfm = KMEANS_DATA['labeled_df']
        df = CLEANING_DATA['clean_df'].copy()
        id_col = 'TransactionID' if 'TransactionID' in df.columns else 'Date'
        df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
        monthly = df.groupby('YearMonth').agg(Transaksi=(id_col, 'nunique'), Revenue=('TotalPrice', 'sum')).reset_index()
        res.update({
            'kpi': {
                'customers': int(rfm['CustomerID'].nunique()),
                'transactions': int(rfm['Frequency'].sum()),
                'revenue': float(rfm['Monetary'].sum()),
                'avg_monetary': float(rfm['Monetary'].mean()),
            },
            'composition': to_json_records(KMEANS_DATA['composition']),
            'monthly': to_json_records(monthly),
            'scatter': KMEANS_DATA['scatter'],
            'profile': to_json_records(KMEANS_DATA['profile']),
            'success': True,
        })
    except Exception as e:
        res['error'] = str(e)
        res['traceback'] = traceback.format_exc()
    DASHBOARD_DATA = res
    return res


# ============================================================
# JAWABAN PERTANYAAN LEMBAR KERJA
# ============================================================
ANSWERS = {
    'cleaning': ("Mengapa data cleaning harus dilakukan sebelum perhitungan RFM dan clustering?",
                 "RFM dan K-Means sepenuhnya dihitung dari angka transaksi. Duplikat menggandakan Frequency dan Monetary, "
                 "tanggal salah merusak Recency, dan harga/jumlah tidak valid mengubah Monetary. K-Means juga berbasis jarak sehingga "
                 "sangat sensitif terhadap outlier dan nilai keliru: pelanggan bisa masuk cluster yang salah dan strategi marketing "
                 "menjadi salah sasaran (garbage in, garbage out)."),
    'rfm': ("Apa dampaknya jika transaksi duplikat tidak dihapus terhadap Frequency dan Monetary?",
            "Frequency akan menggembung karena satu transaksi dihitung berkali-kali, dan Monetary ikut berlipat. Pelanggan biasa tampak "
            "seperti pelanggan loyal bernilai tinggi, total revenue dilaporkan terlalu besar, dan anggaran promosi bisa dialokasikan ke "
            "pelanggan yang sebenarnya tidak bernilai tinggi."),
    'prep': ("Mengapa Frequency dan Monetary cenderung membutuhkan transformasi log?",
             "Keduanya berdistribusi miring ke kanan: sebagian besar pelanggan bernilai kecil, sedikit pelanggan bernilai sangat besar. "
             "Tanpa log, outlier mendominasi jarak Euclidean dan centroid K-Means. Log (LN(x+1)) memampatkan rentang nilai sehingga "
             "distribusi lebih simetris, lalu standardisasi menyamakan skala ketiga fitur agar kontribusinya seimbang."),
    'elbow': ("Apa risiko jika memilih terlalu sedikit atau terlalu banyak cluster?",
              "Terlalu sedikit cluster mencampur pelanggan berbeda perilaku sehingga strategi terlalu umum. Terlalu banyak cluster "
              "memecah pelanggan menjadi kelompok kecil yang mirip (overfitting), sulit diberi makna bisnis, dan membuat kampanye "
              "mahal serta rumit dikelola."),
    'kmeans': ("Jelaskan karakteristik setiap kelompok.",
               "Loyal: Recency rendah, Frequency dan Monetary tinggi. New: Recency rendah tetapi Frequency dan Monetary relatif rendah. "
               "Churn Risk: Recency tinggi dengan Frequency/Monetary relatif rendah. Angka rata-rata tiap cluster tampil pada tabel profil di atas."),
    'dashboard': ("Bagaimana hasil segmentasi dapat mengurangi pemborosan anggaran pemasaran?",
                  "Promosi tidak lagi dikirim sama rata. Loyal cukup dijaga dengan reward non-diskon, New diarahkan ke onboarding "
                  "dan repeat order, Churn Risk diberi penawaran win-back berbatas waktu. Budget dialihkan dari iklan massal ke "
                  "segmen dengan dampak revenue tertinggi dan diukur ROI-nya per segmen."),
}


# ============================================================
# LOGIN TEMPLATE
# ============================================================
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Masuk · TokoKita Analytics</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--ink:#0f2a2e;--teal:#0f766e;--teal2:#0d9488;--mist:#eef5f5;--line:rgba(15,118,110,.16)}
*{-webkit-font-smoothing:antialiased;box-sizing:border-box}
body{font-family:'Plus Jakarta Sans',sans-serif;background:var(--mist);color:var(--ink);margin:0;min-height:100vh}
.wrap{display:flex;min-height:100vh;padding:24px;align-items:center;justify-content:center}
.card{display:grid;grid-template-columns:1fr 1fr;width:100%;max-width:1100px;min-height:600px;border-radius:28px;overflow:hidden;background:#fff;box-shadow:0 30px 70px -24px rgba(15,60,64,.35),0 0 0 1px var(--line)}
.left{position:relative;background:linear-gradient(145deg,#0b2a30 0%,#0f4a4f 55%,#0f766e 100%);display:flex;align-items:center;padding:48px;color:#fff;overflow:hidden}
.left img.bg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:.25}
.left .in{position:relative;z-index:2}
.badge{display:inline-flex;gap:8px;align-items:center;background:rgba(255,255,255,.14);padding:8px 14px;border-radius:20px;font-size:12px;font-weight:700;margin-bottom:28px;border:1px solid rgba(255,255,255,.22)}
.h1{font-size:46px;font-weight:800;line-height:1.08;letter-spacing:-1px;margin:0 0 12px}
.h2{font-size:20px;font-weight:600;color:#a7e3dc;margin:0 0 6px}
.h3{font-size:14px;color:#cfeeea;margin:0}
.corner{position:absolute;top:22px;left:22px;z-index:3;display:flex;gap:10px;align-items:center;background:rgba(255,255,255,.14);padding:6px 14px 6px 6px;border-radius:30px;font-size:11px;font-weight:800}
.corner img{width:30px;height:30px;border-radius:50%;background:#fff;padding:3px;object-fit:contain}
.right{display:flex;align-items:center;justify-content:center;padding:44px 36px;background:linear-gradient(160deg,#fff,#f0f7f6)}
.form{width:100%;max-width:360px}
.photo{width:92px;height:92px;border-radius:22px;object-fit:cover;border:4px solid #fff;box-shadow:0 14px 34px -12px rgba(15,118,110,.45)}
.title{font-size:26px;font-weight:800;color:var(--teal);margin:14px 0 4px}
.inp{width:100%;padding:14px 16px 14px 46px;border:2px solid var(--line);border-radius:14px;font-size:14px;font-family:inherit;outline:none;background:#fff;transition:.2s}
.inp:focus{border-color:var(--teal2);box-shadow:0 0 0 4px rgba(13,148,136,.14)}
.ic{position:absolute;left:16px;top:50%;transform:translateY(-50%);color:var(--teal2)}
.btn{width:100%;padding:15px;border:0;border-radius:14px;background:linear-gradient(135deg,#0f766e,#0e7490);color:#fff;font-weight:700;font-size:15px;cursor:pointer;font-family:inherit;box-shadow:0 12px 28px -12px rgba(15,118,110,.6);transition:.2s}
.btn:hover{transform:translateY(-2px)}
.flash{padding:12px 16px;border-radius:12px;margin-bottom:14px;font-size:13px;font-weight:600}
.flash.error{background:#fde8ea;color:#9f1f33;border-left:4px solid #e36f7f}
.flash.success{background:#dcf5f1;color:#0b5d56;border-left:4px solid #0d9488}
.cred{display:inline-block;padding:5px 12px;border-radius:16px;background:#e1f3f1;color:#0f766e;font-size:11px;font-weight:700;margin:2px}
@media(max-width:900px){.wrap{padding:0}.card{grid-template-columns:1fr;border-radius:0;min-height:100vh}.left{min-height:260px;padding:28px}.h1{font-size:32px}}
</style>
</head>
<body>
<div class="wrap"><div class="card">
  <div class="left">
    <img class="bg" src="/static/download.png" alt="" onerror="this.style.display='none'">
    <div class="corner"><img src="/static/logo unimor.png" alt="" onerror="this.style.display='none'"><span>UNIMOR</span></div>
    <div class="in">
      <div class="badge"><i class="fas fa-users-gear"></i> TokoKita Analytics</div>
      <h1 class="h1">Segmentasi<br>Pelanggan</h1>
      <h2 class="h2">RFM + K-Means Clustering</h2>
      <p class="h3">Data Cleaning, Elbow Method, dan Dashboard BI</p>
    </div>
  </div>
  <div class="right"><div class="form">
    <div style="text-align:center;margin-bottom:24px">
      <img class="photo" src="/static/adelll.JPEG" alt="Foto"
        onerror="this.onerror=null;this.src='data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%230f766e%22 width=%22100%22 height=%22100%22/><text x=%2250%22 y=%2265%22 font-size=%2240%22 fill=%22white%22 text-anchor=%22middle%22 font-weight=%22bold%22>AM</text></svg>'">
      <h1 class="title">Selamat Datang</h1>
      <p style="font-size:13px;color:#0f766e;font-weight:600;margin:0">Customer Segmentation TokoKita</p>
      <p style="font-size:11px;color:#64748b;margin:4px 0 0">Adelia Mutiara · 51230004</p>
    </div>
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% for category, message in messages %}<div class="flash {{ category }}">{{ message }}</div>{% endfor %}
    {% endwith %}
    <form method="POST" action="/login" autocomplete="off">
      <div style="position:relative;margin-bottom:14px"><input class="inp" type="text" name="username" placeholder="Username" required autofocus><i class="fas fa-user ic"></i></div>
      <div style="position:relative;margin-bottom:18px"><input class="inp" type="password" name="password" placeholder="Password" required><i class="fas fa-lock ic"></i></div>
      <button class="btn" type="submit"><i class="fas fa-right-to-bracket" style="margin-right:8px"></i>Masuk ke Dashboard</button>
    </form>
    <div style="text-align:center;margin-top:20px">
      <p style="font-size:10px;color:#64748b;font-weight:700;margin:0 0 6px">AKUN DEMO</p>
      <span class="cred">adel</span><span class="cred">adelia123</span>
      <p style="font-size:10px;color:#94a3b8;margin-top:18px">© 2026 Universitas Timor</p>
    </div>
  </div></div>
</div></div>
</body>
</html>
"""


# ============================================================
# MAIN TEMPLATE
# ============================================================
BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TokoKita Analytics · Adelia Mutiara</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#0f2a2e; --ink2:#2f4b50; --muted:#64807f;
  --teal:#0f766e; --teal2:#0d9488; --teal3:#5eead4; --cyan:#0e7490;
  --mist:#eef5f5; --soft:#e3f1ef; --line:rgba(15,118,110,.16);
  --loyal:#0d9488; --new:#5b8def; --churn:#e36f7f; --amber:#e0a458;
}
*{-webkit-font-smoothing:antialiased}
body{font-family:'Plus Jakarta Sans',sans-serif;color:var(--ink);min-height:100vh;
  background:linear-gradient(135deg,#eef6f5 0%,#e6f1f2 35%,#eaeff8 70%,#eef6f5 100%);background-size:300% 300%;animation:bg 28s ease infinite}
@keyframes bg{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
.mono{font-family:'JetBrains Mono',monospace}
.side{background:linear-gradient(180deg,#ffffff 0%,#f0f8f7 100%);border:1px solid var(--line)}
.glass{background:rgba(255,255,255,.82);backdrop-filter:blur(18px);border:1px solid var(--line);box-shadow:0 4px 24px -10px rgba(15,60,64,.18)}
.nav{display:flex;align-items:center;gap:10px;padding:9px 12px;margin:2px 4px;border-radius:12px;font-size:12px;font-weight:600;color:var(--ink2);position:relative;transition:.2s}
.nav:hover{background:rgba(13,148,136,.08);transform:translateX(3px)}
.nav.on{background:linear-gradient(90deg,rgba(13,148,136,.16),rgba(14,116,144,.06));color:var(--ink)}
.nav.on::before{content:'';position:absolute;left:0;top:20%;bottom:20%;width:4px;border-radius:0 4px 4px 0;background:linear-gradient(180deg,var(--teal2),var(--cyan))}
.nav .ico{width:28px;height:28px;border-radius:9px;display:flex;align-items:center;justify-content:center;background:rgba(13,148,136,.12);color:var(--teal);font-size:11px}
.nav.on .ico{background:var(--teal);color:#fff}
.hero{background:linear-gradient(135deg,#0b2a30 0%,#0f4a4f 45%,#0f766e 80%,#2aa89b 100%);color:#fff;border-radius:26px;padding:32px;position:relative;overflow:hidden;box-shadow:0 24px 54px -22px rgba(15,70,74,.6);margin-bottom:22px}
.hero::after{content:'';position:absolute;top:-45%;right:-12%;width:520px;height:520px;border-radius:50%;background:radial-gradient(circle,rgba(255,255,255,.13),transparent 70%)}
.hero>*{position:relative;z-index:1}
.kpi{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.22);border-radius:16px;padding:16px 18px}
.kpi-card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:18px 20px;display:flex;gap:14px;align-items:center;box-shadow:0 6px 20px -12px rgba(15,60,64,.25)}
.kpi-ic{width:46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:18px;flex-shrink:0}
.sec{background:#fff;border:1px solid var(--line);border-radius:20px;padding:22px;margin-bottom:18px;position:relative;box-shadow:0 8px 26px -16px rgba(15,60,64,.25)}
.sec::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;border-radius:20px 20px 0 0;background:linear-gradient(90deg,var(--teal),var(--teal2),var(--cyan))}
.sec-h{display:flex;align-items:center;gap:12px;padding-bottom:14px;margin-bottom:16px;border-bottom:1px dashed var(--line)}
.sec-i{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:17px;flex-shrink:0}
.sec-t{font-weight:800;font-size:16px;color:var(--ink)}
.sec-s{font-size:11px;color:var(--muted)}
.tbl-wrap{overflow:auto;border-radius:14px;border:1px solid var(--line);max-height:520px}
table.t{width:100%;border-collapse:collapse;font-size:12px}
table.t thead th{position:sticky;top:0;background:linear-gradient(90deg,var(--teal),var(--cyan));color:#fff;text-align:left;padding:10px 12px;font-size:11px;font-weight:700;white-space:nowrap}
table.t td{padding:9px 12px;border-top:1px solid rgba(15,118,110,.09);white-space:nowrap;color:var(--ink2)}
table.t tbody tr:hover{background:rgba(13,148,136,.06)}
.btn{display:inline-flex;align-items:center;gap:8px;font-weight:700;padding:13px 26px;border-radius:14px;color:#fff;background:linear-gradient(135deg,var(--teal),var(--cyan));box-shadow:0 12px 26px -14px rgba(15,118,110,.7);border:0;cursor:pointer;font-family:inherit;transition:.2s}
.btn:hover{transform:translateY(-2px)}
.btn-light{background:#fff;color:var(--teal);box-shadow:0 8px 20px -10px rgba(0,0,0,.25)}
.cta{background:linear-gradient(135deg,#0f4a4f,#0f766e);border-radius:20px;padding:22px;text-align:center;margin-top:8px}
.stat{border-radius:16px;padding:16px;color:#fff;position:relative;overflow:hidden}
.chip{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:14px;font-size:11px;font-weight:700}
.chip-a{background:rgba(224,164,88,.2);color:#8a5a14}.chip-t{background:rgba(13,148,136,.14);color:#0b5d56}.chip-r{background:rgba(227,111,127,.16);color:#a22c40}
.seg{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:16px;font-size:11px;font-weight:800;color:#fff}
.seg-Loyal{background:var(--loyal)}.seg-New{background:var(--new)}.seg-Churn{background:var(--churn)}
.strat{border-radius:16px;padding:18px;border-left:5px solid;margin-bottom:12px;background:#fff}
.strat-Loyal{border-color:var(--loyal);background:linear-gradient(135deg,#e6f6f3,#fff)}
.strat-New{border-color:var(--new);background:linear-gradient(135deg,#eaf1fd,#fff)}
.strat-Churn{border-color:var(--churn);background:linear-gradient(135deg,#fdecee,#fff)}
.qa{background:linear-gradient(135deg,#f0f8f7,#fff);border:1px solid var(--line);border-left:5px solid var(--amber);border-radius:16px;padding:16px 18px;margin-top:16px}
.ch{position:relative;height:340px;width:100%}.ch-tall{position:relative;height:430px;width:100%}
.upload{border:2px dashed rgba(13,148,136,.45);background:linear-gradient(135deg,rgba(13,148,136,.07),rgba(14,116,144,.07));border-radius:20px;transition:.25s}
.upload:hover{border-color:var(--teal);transform:scale(1.004)}
.orb{position:fixed;border-radius:50%;filter:blur(90px);opacity:.22;z-index:0;pointer-events:none}
.photo-s{width:40px;height:40px;border-radius:12px;object-fit:cover;border:2px solid rgba(13,148,136,.4)}
.kinp{width:140px;padding:12px 16px;border:2px solid var(--line);border-radius:14px;font-size:18px;font-weight:800;text-align:center;font-family:inherit;outline:none;background:#fff;color:var(--ink);transition:.2s}
.kinp:focus{border-color:var(--teal2);box-shadow:0 0 0 4px rgba(13,148,136,.14)}
.cnum{display:inline-flex;align-items:center;justify-content:center;min-width:30px;height:30px;padding:0 10px;border-radius:15px;font-size:12px;font-weight:800;color:#fff}
.ket{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:14px;font-size:11px;font-weight:800;color:#fff;white-space:nowrap}
.lv{display:inline-block;padding:2px 9px;border-radius:12px;font-size:10px;font-weight:800}
.lv-Tinggi{background:rgba(13,148,136,.16);color:#0b5d56}.lv-Sedang{background:rgba(224,164,88,.22);color:#8a5a14}.lv-Rendah{background:rgba(227,111,127,.18);color:#a22c40}
</style>
</head>
<body>
<div class="orb w-96 h-96" style="top:-110px;left:-110px;background:#5eead4"></div>
<div class="orb w-80 h-80" style="bottom:-100px;right:-100px;background:#7aa7f0"></div>

<div class="flex h-screen overflow-hidden relative z-10">

<!-- SIDEBAR -->
<aside class="side w-72 m-3 mr-0 rounded-3xl shadow-xl flex flex-col overflow-hidden">
  <div class="p-5 border-b" style="border-color:var(--line)">
    <div class="flex items-center gap-3">
      <div class="w-11 h-11 rounded-2xl flex items-center justify-center shadow-lg" style="background:linear-gradient(135deg,#0d9488,#0e7490)"><i class="fas fa-users-gear text-white text-lg"></i></div>
      <div><h1 class="text-base font-extrabold" style="color:var(--ink)">TokoKita Analytics</h1><p class="text-[11px] font-semibold" style="color:var(--teal)">RFM + K-Means</p></div>
    </div>
  </div>
  <div class="mx-4 my-3 p-3 rounded-2xl" style="background:var(--soft);border:1px solid var(--line)">
    <div class="flex items-center gap-3">
      <img src="/static/adelll.JPEG" alt="Foto" class="photo-s"
        onerror="this.onerror=null;this.src='data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%230f766e%22 width=%22100%22 height=%22100%22/><text x=%2250%22 y=%2265%22 font-size=%2240%22 fill=%22white%22 text-anchor=%22middle%22 font-weight=%22bold%22>AM</text></svg>'">
      <div class="min-w-0"><p class="text-xs font-bold truncate">ADELIA MUTIARA</p><p class="text-[10px] mono" style="color:var(--teal)">NPM: 51230004</p></div>
    </div>
  </div>
  <nav class="flex-1 px-2 overflow-y-auto">
  {% if sheets %}
    <p class="px-3 py-2 text-[11px] font-bold" style="color:var(--muted)">Alur analisis</p>
    {% for key, label, icon, done in nav_main %}
    <a href="/?view={{ key }}" class="nav {{ 'on' if active_view == key else '' }}">
      <span class="ico"><i class="fas {{ icon }}"></i></span><span class="flex-1">{{ label }}</span>
      {% if done %}<i class="fas fa-circle-check text-[11px]" style="color:var(--teal2)"></i>{% endif %}
    </a>
    {% endfor %}
    <p class="px-3 py-2 mt-3 text-[11px] font-bold" style="color:var(--muted)">Hasil &amp; laporan</p>
    {% for key, label, icon, done in nav_out %}
    <a href="/?view={{ key }}" class="nav {{ 'on' if active_view == key else '' }}">
      <span class="ico"><i class="fas {{ icon }}"></i></span><span class="flex-1">{{ label }}</span>
    </a>
    {% endfor %}
  {% else %}
    <div class="px-3 py-10 text-center">
      <div class="w-20 h-20 mx-auto rounded-3xl flex items-center justify-center mb-4" style="background:var(--soft)"><i class="fas fa-file-arrow-up text-3xl" style="color:var(--teal2)"></i></div>
      <p class="text-xs font-bold mb-1">Belum ada data</p><p class="text-[11px]" style="color:var(--muted)">Unggah file Excel untuk memulai</p>
    </div>
  {% endif %}
  </nav>
  <div class="p-3 border-t space-y-2" style="border-color:var(--line)">
    <a href="/?view=upload" class="btn w-full justify-center !py-2.5 text-xs"><i class="fas fa-plus-circle"></i> Unggah file baru</a>
    <a href="/logout" class="flex items-center justify-center gap-2 w-full px-3 py-2 rounded-xl text-xs font-bold" style="background:rgba(227,111,127,.12);color:#b4344a"><i class="fas fa-sign-out-alt"></i> Keluar</a>
  </div>
</aside>

<!-- MAIN -->
<main class="flex-1 flex flex-col overflow-hidden">
  <header class="m-3 glass rounded-3xl px-6 py-4 flex justify-between items-center">
    <div>
      <p class="text-[11px] font-semibold mb-1" style="color:var(--teal)">Dashboard / {{ page_title }}</p>
      <h2 class="text-xl font-extrabold flex items-center gap-2"><i class="fas {{ page_icon }}" style="color:var(--teal2)"></i>{{ page_title }}</h2>
    </div>
    {% if sheets %}<div class="hidden md:flex items-center gap-2 px-3 py-2 rounded-xl" style="background:var(--soft);border:1px solid var(--line)">
      <i class="fas fa-table text-xs" style="color:var(--teal)"></i><span class="text-[11px] font-bold" style="color:var(--teal)">{{ sheets[0] }}</span></div>{% endif %}
  </header>

  <div class="flex-1 px-3 pb-3 overflow-y-auto">
  {% with messages = get_flashed_messages(with_categories=true) %}
    {% for category, message in messages %}
    <div class="mb-3 p-4 rounded-2xl glass flex items-center gap-3 text-sm font-semibold" style="border-left:4px solid {{ '#e36f7f' if category == 'error' else '#0d9488' }};color:{{ '#a22c40' if category == 'error' else '#0b5d56' }}">
      <i class="fas {{ 'fa-circle-exclamation' if category == 'error' else 'fa-circle-check' }}"></i><span>{{ message }}</span></div>
    {% endfor %}
  {% endwith %}

{% macro table(tb) %}
  <div class="tbl-wrap"><table class="t"><thead><tr>{% for h in tb[0] %}<th>{{ h }}</th>{% endfor %}</tr></thead>
  <tbody>{% for r in tb[1] %}<tr>{% for c in r %}<td>{{ c }}</td>{% endfor %}</tr>{% endfor %}</tbody></table></div>
{% endmacro %}
{% macro sech(icon, color, title, sub) %}
  <div class="sec-h"><div class="sec-i" style="background:{{ color }}"><i class="fas {{ icon }}"></i></div>
  <div class="flex-1"><div class="sec-t">{{ title }}</div><div class="sec-s">{{ sub }}</div></div></div>
{% endmacro %}
{% macro segbadge(label) %}
  {% if label == 'Loyal' %}<span class="seg seg-Loyal"><i class="fas fa-crown"></i> Loyal</span>
  {% elif label == 'New' %}<span class="seg seg-New"><i class="fas fa-seedling"></i> New</span>
  {% else %}<span class="seg seg-Churn"><i class="fas fa-triangle-exclamation"></i> Churn Risk</span>{% endif %}
{% endmacro %}
{% macro qa(key) %}
  <div class="qa"><p class="text-xs font-extrabold mb-1" style="color:#8a5a14"><i class="fas fa-pen-to-square mr-1"></i> Pertanyaan: {{ answers[key][0] }}</p>
  <p class="text-xs font-bold mb-1" style="color:var(--ink)">Jawaban / catatan:</p><p class="text-xs leading-relaxed" style="color:var(--ink2)">{{ answers[key][1] }}</p></div>
{% endmacro %}
{% macro gate(msg, view, btn) %}
  <div class="glass rounded-3xl p-8 text-center max-w-xl mx-auto"><p class="text-sm mb-4" style="color:var(--ink2)">{{ msg }}</p>
  <a href="/?view={{ view }}" class="btn"><i class="fas fa-arrow-left"></i> {{ btn }}</a></div>
{% endmacro %}
{% macro runcard(icon, title, desc, action, btn, hidden='') %}
  <div class="max-w-2xl mx-auto glass rounded-3xl p-8 text-center shadow-xl">
    <div class="w-20 h-20 mx-auto rounded-3xl flex items-center justify-center mb-4 shadow-xl" style="background:linear-gradient(135deg,#0d9488,#0e7490)"><i class="fas {{ icon }} text-3xl text-white"></i></div>
    <h3 class="text-2xl font-extrabold mb-2">{{ title }}</h3><p class="text-sm mb-6" style="color:var(--ink2)">{{ desc }}</p>
    <form action="{{ action }}" method="POST">{{ hidden|safe }}<button class="btn" type="submit"><i class="fas fa-play"></i> {{ btn }}</button></form>
  </div>
{% endmacro %}
{% macro next(action, label, hidden='') %}
  <div class="cta"><form action="{{ action }}" method="POST">{{ hidden|safe }}<button type="submit" class="btn btn-light"><i class="fas fa-arrow-right"></i> {{ label }}</button></form></div>
{% endmacro %}

{% if not sheets or active_view == 'upload' %}
  <!-- UPLOAD -->
  <div class="max-w-3xl mx-auto">
    <div class="text-center mb-6">
      <div class="w-24 h-24 mx-auto rounded-3xl flex items-center justify-center shadow-2xl mb-4" style="background:linear-gradient(135deg,#0d9488,#0e7490)"><i class="fas fa-users-gear text-4xl text-white"></i></div>
      <h1 class="text-3xl md:text-4xl font-extrabold mb-3">Segmentasi <span style="color:var(--teal2)">Pelanggan</span> TokoKita</h1>
      <p class="text-sm max-w-lg mx-auto" style="color:var(--ink2)">Unggah file Excel. Hanya sheet <b>Data Transaksi Mentah</b> yang dibaca, sheet lain diabaikan.</p>
    </div>
    <div class="glass rounded-3xl shadow-2xl p-6">
      <form action="/upload" method="POST" enctype="multipart/form-data" id="uploadForm">
        <div class="upload relative p-10 text-center cursor-pointer">
          <input type="file" name="file" id="fileInput" accept=".xlsx,.xls" required class="absolute inset-0 w-full h-full opacity-0 cursor-pointer">
          <div class="pointer-events-none">
            <div class="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center mb-4" style="background:rgba(13,148,136,.12)"><i class="fas fa-file-excel text-3xl" style="color:var(--teal)"></i></div>
            <p class="text-base font-bold mb-1">Klik atau seret file ke sini</p>
            <p class="text-xs mb-3" style="color:var(--teal)">Format .xlsx atau .xls, maksimal 100 MB</p>
            <p class="text-[11px] font-semibold" id="fileName" style="color:var(--ink2)"></p>
          </div>
        </div>
        <button type="submit" id="submitBtn" class="btn w-full justify-center mt-5 !py-4"><i class="fas fa-rocket"></i> Unggah &amp; baca data</button>
      </form>
    </div>
  </div>

{% elif active_view == 'sheets' %}
  <!-- PREVIEW DATA MENTAH -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
    <div class="stat" style="background:linear-gradient(135deg,#0d9488,#2aa89b)"><p class="text-[11px] font-bold opacity-90">Sheet dipakai</p><p class="text-sm font-extrabold mt-1">{{ sheets[0] }}</p></div>
    <div class="stat" style="background:linear-gradient(135deg,#0e7490,#22a1bd)"><p class="text-[11px] font-bold opacity-90">Total baris</p><p class="text-2xl font-extrabold">{{ raw_rows }}</p></div>
    <div class="stat" style="background:linear-gradient(135deg,#2f5f8f,#5b8def)"><p class="text-[11px] font-bold opacity-90">Total kolom</p><p class="text-2xl font-extrabold">{{ raw_cols }}</p></div>
    <div class="stat" style="background:linear-gradient(135deg,#4b5d73,#7a8ba3)"><p class="text-[11px] font-bold opacity-90">Status</p><p class="text-lg font-extrabold">Data mentah</p></div>
  </div>
  <div class="sec">
    {{ sech('fa-table', 'linear-gradient(135deg,#0d9488,#0e7490)', 'Pratinjau Data Transaksi Mentah', 'Menampilkan 100 baris pertama, belum dibersihkan') }}
    {{ table(raw_table) }}
  </div>
  <div class="cta"><a href="/?view=cleaning" class="btn btn-light"><i class="fas fa-arrow-right"></i> Lanjut ke Data Cleaning</a></div>

{% elif active_view == 'cleaning' %}
  {% if not cleaning_ok %}
    {{ runcard('fa-broom', 'Data Cleaning', 'Identifikasi duplikat, missing value, dan nilai tidak valid pada Data Transaksi Mentah, lalu hasilkan data bersih.', '/cleaning', 'Proses Data Cleaning') }}
    {% if clean.error %}<div class="glass rounded-2xl p-4 mt-4 max-w-2xl mx-auto text-sm" style="color:#a22c40">{{ clean.error }}</div>{% endif %}
  {% else %}
  <div class="hero">
    <div class="flex items-start justify-between flex-wrap gap-4 mb-5">
      <div><span class="inline-flex items-center gap-2 bg-white/20 px-3 py-1.5 rounded-full text-xs font-bold mb-3"><i class="fas fa-broom"></i> Bagian 0 · Data Cleaning</span>
        <h1 class="text-3xl font-extrabold mb-1">Laporan Data Quality</h1>
        <p class="text-sm opacity-90 max-w-2xl">Sumber: {{ clean.sheet_name }}. Perbandingan kondisi data sebelum dan sesudah cleaning.</p></div>
      <div class="text-right"><p class="text-xs opacity-80 font-bold">Data bersih</p><p class="text-4xl font-extrabold">{{ pct_clean }}%</p><p class="text-xs opacity-80">dari data awal</p></div>
    </div>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Baris sebelum</p><p class="text-2xl font-extrabold">{{ fi(clean.rows_before) }}</p></div>
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Baris sesudah</p><p class="text-2xl font-extrabold">{{ fi(clean.rows_after) }}</p></div>
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Baris dihapus</p><p class="text-2xl font-extrabold">{{ fi(clean.rows_before - clean.rows_after) }}</p></div>
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Pelanggan unik</p><p class="text-2xl font-extrabold">{{ fi(clean.clean_df['CustomerID'].nunique()) }}</p></div>
    </div>
  </div>

  <div class="sec">
    {{ sech('fa-magnifying-glass-chart', 'linear-gradient(135deg,#0d9488,#0e7490)', 'Tugas 0.1 · Identifikasi masalah data', 'Baris, kolom, nilai kosong, duplikat, dan nilai tidak valid: sebelum vs sesudah') }}
    <div class="tbl-wrap mb-4"><table class="t"><thead><tr><th>Metrik</th><th>Sebelum</th><th>Sesudah</th><th>Status</th></tr></thead><tbody>
      {% for q in clean.quality %}<tr><td class="font-semibold">{{ q.metrik }}</td><td class="mono">{{ fi(q.sebelum) }}</td><td class="mono">{{ fi(q.sesudah) }}</td>
        <td>{% if loop.index <= 2 %}<span class="chip chip-t">info</span>{% elif q.sesudah == 0 %}<span class="chip chip-t"><i class="fas fa-check"></i> bersih</span>{% else %}<span class="chip chip-a">cek</span>{% endif %}</td></tr>{% endfor %}
    </tbody></table></div>
    {% if clean.missing_by_col %}
    <p class="text-xs font-bold mb-2">Nilai kosong per kolom (data mentah)</p>
    <div class="flex flex-wrap gap-2">{% for c, n in clean.missing_by_col.items() %}<span class="chip chip-a mono">{{ c }}: {{ n }}</span>{% endfor %}</div>
    {% endif %}
  </div>

  <div class="sec">
    {{ sech('fa-clone', 'linear-gradient(135deg,#e36f7f,#ee98a3)', 'Tugas 0.2 · Duplikat', 'Duplikat persis dihapus; duplikat transaksi (ID beda, isi sama) diidentifikasi dan dihapus') }}
    <div class="grid grid-cols-2 gap-3 mb-4">
      <div class="stat" style="background:linear-gradient(135deg,#d85d70,#ee98a3)"><p class="text-[11px] font-bold opacity-90">Duplikat persis dihapus</p><p class="text-3xl font-extrabold">{{ fi(clean.n_dup_exact) }}</p></div>
      <div class="stat" style="background:linear-gradient(135deg,#c98a3e,#e0a458)"><p class="text-[11px] font-bold opacity-90">Duplikat transaksi (ID beda)</p><p class="text-3xl font-extrabold">{{ fi(clean.n_dup_trx) }}</p></div>
    </div>
    {% if clean.dup_examples %}<p class="text-xs font-bold mb-2">Contoh duplikat persis</p>{{ table(clean.dup_examples) }}{% endif %}
    {% if clean.dup_trx_examples %}<p class="text-xs font-bold mt-4 mb-2">Contoh duplikat transaksi (pelanggan, tanggal, produk, qty, harga sama)</p>{{ table(clean.dup_trx_examples) }}{% endif %}
  </div>

  <div class="sec">
    {{ sech('fa-circle-question', 'linear-gradient(135deg,#c98a3e,#e0a458)', 'Tugas 0.3 · Missing value', 'Keputusan penanganan beserta alasan bisnis') }}
    <div class="tbl-wrap"><table class="t"><thead><tr><th>Kolom</th><th>Jumlah kosong</th><th>Keputusan</th><th>Alasan bisnis</th></tr></thead><tbody>
      {% for m in clean.missing_report %}<tr><td class="mono font-bold">{{ m.kolom }}</td><td class="mono">{{ m.n }}</td><td>{{ m.metode }}</td><td style="white-space:normal">{{ m.alasan }}</td></tr>{% endfor %}
    </tbody></table></div>
  </div>

  <div class="sec">
    {{ sech('fa-text-height', 'linear-gradient(135deg,#0e7490,#22a1bd)', 'Tugas 0.4 · Standarisasi', 'Kategori, nama pelanggan, tanggal, dan format angka') }}
    <div class="tbl-wrap"><table class="t"><thead><tr><th>Kolom</th><th>Aksi</th><th>Detail</th></tr></thead><tbody>
      {% for r in clean.standardization_log %}<tr><td class="font-bold">{{ r.kolom }}</td><td>{{ r.aksi }}</td><td style="white-space:normal">{{ r.detail }}</td></tr>{% endfor %}
    </tbody></table></div>
  </div>

  <div class="sec">
    {{ sech('fa-ban', 'linear-gradient(135deg,#5b8def,#85abf5)', 'Tugas 0.5 · Nilai tidak valid', 'Baris bermasalah yang dibuang atau dihitung ulang') }}
    <div class="tbl-wrap"><table class="t"><thead><tr><th>Jenis masalah</th><th>Jumlah baris</th></tr></thead><tbody>
      {% for r in clean.invalid_log %}<tr><td>{{ r.jenis }}</td><td class="mono font-bold">{{ fi(r.n) }}</td></tr>{% endfor %}
    </tbody></table></div>
  </div>

  <div class="sec">
    {{ sech('fa-sparkles', 'linear-gradient(135deg,#0d9488,#2aa89b)', 'Tugas 0.6 · Dataset transaksi bersih', 'Sumber analisis RFM, menampilkan 15 baris pertama') }}
    {{ table(clean.clean_table) }}
    <div class="mt-4 text-center"><a href="/download/clean" class="btn"><i class="fas fa-download"></i> Unduh dataset bersih (CSV)</a></div>
    {{ qa('cleaning') }}
  </div>
  {{ next('/rfm', 'Lanjut ke Perhitungan RFM') }}
  {% endif %}

{% elif active_view == 'rfm' %}
  {% if not cleaning_ok %}{{ gate('Lakukan Data Cleaning terlebih dahulu.', 'cleaning', 'Ke Data Cleaning') }}
  {% elif not rfm_ok %}{{ runcard('fa-chart-simple', 'Perhitungan RFM', 'Recency = 30 Juni 2024 dikurangi transaksi terakhir. Frequency = jumlah transaksi valid. Monetary = total nilai transaksi valid.', '/rfm', 'Hitung RFM') }}
  {% else %}
  <div class="hero">
    <div class="flex items-start justify-between flex-wrap gap-4 mb-5">
      <div><span class="inline-flex items-center gap-2 bg-white/20 px-3 py-1.5 rounded-full text-xs font-bold mb-3"><i class="fas fa-chart-simple"></i> Bagian 1 · RFM</span>
        <h1 class="text-3xl font-extrabold mb-1">Tabel RFM</h1><p class="text-sm opacity-90">Satu baris untuk setiap pelanggan, tanggal acuan 30 Juni 2024.</p></div>
      <div class="text-right"><p class="text-xs opacity-80 font-bold">Total pelanggan</p><p class="text-4xl font-extrabold">{{ fi(rfm.n_customers) }}</p></div>
    </div>
    <div class="grid grid-cols-3 gap-3">
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Rata-rata Recency</p><p class="text-2xl font-extrabold">{{ '%.1f'|format(rfm.avg_recency) }} <span class="text-xs">hari</span></p></div>
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Rata-rata Frequency</p><p class="text-2xl font-extrabold">{{ '%.2f'|format(rfm.avg_frequency) }}</p></div>
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Rata-rata Monetary</p><p class="text-lg font-extrabold">{{ rp(rfm.avg_monetary) }}</p></div>
    </div>
  </div>
  <div class="sec">
    {{ sech('fa-table', 'linear-gradient(135deg,#0d9488,#0e7490)', 'Tabel RFM per pelanggan', 'Template: ID_Pelanggan sampai Segmen_Bisnis, 50 baris pertama') }}
    <div class="qa" style="border-left-color:var(--teal2);margin-top:0;margin-bottom:16px">
      <p class="text-xs font-extrabold mb-2" style="color:var(--teal)"><i class="fas fa-circle-info mr-1"></i> Petunjuk</p>
      <ul class="text-xs space-y-1 ml-5 list-disc" style="color:var(--ink2)">
        <li>Gunakan data transaksi yang telah dibersihkan.</li>
        <li>Recency dihitung dari 30 Juni 2024.</li>
        <li>Log Frequency = LN(Frequency+1).</li>
        <li>Log Monetary = LN(Monetary+1).</li>
        <li>Standardisasi ketiga fitur hasil preprocessing.</li>
      </ul>
      <p class="text-[11px] mt-2" style="color:var(--muted)">Seluruh kolom dihitung otomatis dari data bersih, termasuk Log, Z, Cluster, dan Segmen_Bisnis (jumlah cluster mengikuti k optimal dari Elbow Method). Langkah detail tetap bisa dilihat di menu Transformasi, Elbow Method, dan K-Means.</p>
    </div>
    {{ table(rfm_tpl) }}
    <p class="text-[11px] mt-2 text-center" style="color:var(--muted)"><i class="fas fa-circle-info mr-1"></i>Menampilkan 50 dari {{ fi(rfm.n_customers) }} pelanggan. Untuk melihat semua data, gunakan tombol unduh.</p>
    <div class="mt-3 text-center"><a href="/download/rfm" class="btn"><i class="fas fa-download"></i> Unduh tabel RFM (CSV)</a></div>
    {{ qa('rfm') }}
  </div>
  {{ next('/prep', 'Lanjut ke Transformasi') }}
  {% endif %}

{% elif active_view == 'prep' %}
  {% if not rfm_ok %}{{ gate('Lakukan Perhitungan RFM terlebih dahulu.', 'rfm', 'Ke Perhitungan RFM') }}
  {% elif not prep_ok %}{{ runcard('fa-wand-magic-sparkles', 'Transformasi & Standardisasi', 'Log_Frequency = LN(Frequency + 1), Log_Monetary = LN(Monetary + 1), lalu standardisasi Recency, Log_Frequency, dan Log_Monetary.', '/prep', 'Proses Transformasi') }}
  {% else %}
  <div class="hero">
    <span class="inline-flex items-center gap-2 bg-white/20 px-3 py-1.5 rounded-full text-xs font-bold mb-3"><i class="fas fa-wand-magic-sparkles"></i> Bagian 2 · Transformasi</span>
    <h1 class="text-3xl font-extrabold mb-1">Log Transformation &amp; Standardisasi</h1>
    <p class="text-sm opacity-90 mb-5">Setelah standardisasi, setiap fitur memiliki mean mendekati 0 dan standar deviasi mendekati 1.</p>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
      {% for s in prep.stats %}<div class="kpi"><p class="text-[11px] font-bold opacity-80">{{ s.fitur }}</p><p class="text-xs opacity-90 mono">mean = {{ '%.4f'|format(s.mean) }}</p><p class="text-xs opacity-90 mono">std = {{ '%.4f'|format(s.std) }}</p></div>{% endfor %}
    </div>
  </div>
  <div class="sec">
    {{ sech('fa-chart-area', 'linear-gradient(135deg,#5b8def,#85abf5)', 'Efek transformasi log pada kemiringan (skewness)', 'Semakin dekat 0, distribusi semakin simetris') }}
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
      {% for k, v in prep.skew.items() %}<div class="stat" style="background:linear-gradient(135deg,{{ '#0e7490,#22a1bd' if 'Log' in k else '#4b5d73,#7a8ba3' }})"><p class="text-[11px] font-bold opacity-90">Skewness {{ k }}</p><p class="text-2xl font-extrabold mono">{{ '%.2f'|format(v) }}</p></div>{% endfor %}
    </div>
  </div>
  <div class="sec">
    {{ sech('fa-table', 'linear-gradient(135deg,#0d9488,#0e7490)', 'Hasil transformasi dan standardisasi', '50 baris pertama') }}
    {{ table(prep.table) }}
    {{ qa('prep') }}
  </div>
  {{ next('/elbow', 'Lanjut ke Elbow Method') }}
  {% endif %}

{% elif active_view == 'elbow' %}
  {% if not prep_ok %}{{ gate('Lakukan Transformasi terlebih dahulu.', 'prep', 'Ke Transformasi') }}
  {% elif not elbow_ok %}{{ runcard('fa-chart-line', 'Elbow Method', 'Jalankan K-Means untuk k = 2, 3, 4, 5, 6 dan catat inertia/WCSS.', '/elbow', 'Jalankan Elbow Method') }}
  {% else %}
  <div class="hero">
    <div class="flex items-start justify-between flex-wrap gap-4">
      <div><span class="inline-flex items-center gap-2 bg-white/20 px-3 py-1.5 rounded-full text-xs font-bold mb-3"><i class="fas fa-chart-line"></i> Bagian 3 · Elbow Method</span>
        <h1 class="text-3xl font-extrabold mb-1">Menentukan jumlah cluster</h1><p class="text-sm opacity-90">Titik siku dipilih dari jarak terjauh kurva inertia ke garis lurus k=2 sampai k=6.</p></div>
      <div class="text-right"><p class="text-xs opacity-80 font-bold">k optimal</p><p class="text-5xl font-extrabold">{{ elbow.optimal_k }}</p></div>
    </div>
  </div>
  <div class="sec">
    {{ sech('fa-table', 'linear-gradient(135deg,#0d9488,#0e7490)', 'Tabel inertia k = 2 sampai 6', 'WCSS makin kecil seiring k bertambah; cari titik penurunan melandai') }}
    <div class="tbl-wrap mb-5"><table class="t"><thead><tr><th>k</th><th>Inertia / WCSS</th><th>Penurunan dari k sebelumnya</th><th>Status</th></tr></thead><tbody>
      {% for k in elbow.k_values %}<tr style="{{ 'background:rgba(13,148,136,.10)' if k == elbow.optimal_k else '' }}">
        <td class="font-bold">{{ k }}</td><td class="mono">{{ '%.2f'|format(elbow.inertia[loop.index0]) }}</td>
        <td class="mono">{% if loop.index0 == 0 %}-{% else %}{{ '%.1f'|format(elbow.drops[loop.index0 - 1]) }}%{% endif %}</td>
        <td>{% if k == elbow.optimal_k %}<span class="chip chip-t"><i class="fas fa-star"></i> k optimal</span>{% else %}-{% endif %}</td></tr>{% endfor %}
    </tbody></table></div>
    <div class="ch"><canvas id="chart_elbow"></canvas></div>
    <div class="qa" style="border-left-color:var(--teal2)"><p class="text-xs font-extrabold mb-1">Alasan memilih k = {{ elbow.optimal_k }}</p>
      <p class="text-xs leading-relaxed" style="color:var(--ink2)"><b>Statistik:</b> pada k = {{ elbow.optimal_k }} kurva inertia membentuk siku, penurunan WCSS sesudahnya melandai sehingga tambahan cluster memberi perbaikan kecil. <b>Bisnis:</b> segmen Loyal, New, dan Churn Risk cukup untuk diberi strategi berbeda tanpa membuat kampanye terlalu rumit.</p></div>
    {{ qa('elbow') }}
  </div>
  {{ next('/kmeans', 'Lanjut ke K-Means (k = ' ~ elbow.optimal_k ~ ')', '<input type="hidden" name="k" value="' ~ elbow.optimal_k ~ '">') }}
  <script>
  document.addEventListener('DOMContentLoaded',()=>{
    const ks={{ elbow.k_values|tojson }}, inertia={{ elbow.inertia|tojson }}, opt={{ elbow.optimal_k }};
    new Chart(document.getElementById('chart_elbow'),{type:'line',
      data:{labels:ks.map(k=>'k = '+k),datasets:[{label:'Inertia',data:inertia,borderColor:'#0f766e',backgroundColor:'rgba(13,148,136,.14)',borderWidth:3,fill:true,tension:.3,
        pointBackgroundColor:ks.map(k=>k===opt?'#e36f7f':'#0d9488'),pointBorderColor:'#fff',pointBorderWidth:2,pointRadius:ks.map(k=>k===opt?10:6)}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
        scales:{x:{grid:{display:false}},y:{title:{display:true,text:'Inertia / WCSS'},grid:{color:'rgba(15,118,110,.08)'}}}}});
  });
  </script>
  {% endif %}

{% elif active_view == 'kmeans' %}
  {% if not elbow_ok %}{{ gate('Lakukan Elbow Method terlebih dahulu.', 'elbow', 'Ke Elbow Method') }}
  {% elif not kmeans_ok %}{{ runcard('fa-object-group', 'K-Means Clustering', 'Jalankan K-Means dengan k optimal = ' ~ elbow.optimal_k ~ ' sampai konvergen.', '/kmeans', 'Jalankan K-Means', '<input type="hidden" name="k" value="' ~ elbow.optimal_k ~ '">') }}
  {% else %}
  <div class="hero">
    <span class="inline-flex items-center gap-2 bg-white/20 px-3 py-1.5 rounded-full text-xs font-bold mb-3"><i class="fas fa-object-group"></i> Bagian 4 · K-Means</span>
    <h1 class="text-3xl font-extrabold mb-1">K-Means &amp; Interpretasi</h1>
    <p class="text-sm opacity-90">k = {{ kmeans.k }}, konvergen dalam {{ kmeans.iterations }} iterasi. Label bisnis ditentukan dari profil RFM tiap cluster.</p>
  </div>
  <div class="sec">
    {{ sech('fa-table', 'linear-gradient(135deg,#0d9488,#0e7490)', 'Profil cluster', 'Rata-rata Recency, Frequency, Monetary tiap cluster dan label bisnis') }}
    {{ profile_table(kmeans.profile, False) }}
    <div class="ch mt-5"><canvas id="chart_profile"></canvas></div>
  </div>
  <div class="sec">
    {{ sech('fa-braille', 'linear-gradient(135deg,#5b8def,#85abf5)', 'Visualisasi cluster', 'Frequency vs Monetary (sampel hingga 800 pelanggan)') }}
    <div class="ch-tall"><canvas id="chart_scatter"></canvas></div>
  </div>
  <div class="sec">
    {{ sech('fa-list-check', 'linear-gradient(135deg,#c98a3e,#e0a458)', 'Karakteristik tiap kelompok', '') }}
    {% for r in kmeans.profile.to_dict('records') %}
      <div class="strat {{ 'strat-Loyal' if r.Label_Bisnis == 'Loyal' else ('strat-New' if r.Label_Bisnis == 'New' else 'strat-Churn') }}">
        <div class="flex items-center gap-3 mb-2">{{ segbadge(r.Label_Bisnis) }}<span class="text-xs font-semibold" style="color:var(--muted)">Cluster {{ r.Cluster }} · {{ fi(r.Jumlah_Pelanggan) }} pelanggan</span></div>
        <p class="text-xs" style="color:var(--ink2)">{{ r.Karakteristik }}</p></div>
    {% endfor %}
    <p class="text-xs font-bold mt-5 mb-2">Pelanggan + label segmen (50 baris pertama)</p>
    {{ table(kmeans.table) }}
    <p class="text-[11px] mt-2 text-center" style="color:var(--muted)"><i class="fas fa-circle-info mr-1"></i>Menampilkan 50 dari {{ fi(kmeans.labeled_df|length) }} pelanggan. Untuk melihat semua data, gunakan tombol unduh.</p>
    <div class="mt-3 text-center"><a href="/download/segment" class="btn"><i class="fas fa-download"></i> Unduh pelanggan + label (CSV)</a></div>
    {{ qa('kmeans') }}
  </div>
  {{ next('/dashboard', 'Lihat Dashboard BI') }}
  <script>
  const COL={'Loyal':'#0d9488','New':'#5b8def','Churn Risk':'#e36f7f'};
  document.addEventListener('DOMContentLoaded',()=>{
    const prof={{ kmeans.profile.to_dict('records')|tojson }};
    const lbl=prof.map(p=>p.Label_Bisnis+' (C'+p.Cluster+')');
    new Chart(document.getElementById('chart_profile'),{type:'bar',data:{labels:lbl,datasets:[
      {label:'Avg Recency (hari)',data:prof.map(p=>p.Avg_Recency),backgroundColor:'#0e7490',borderRadius:8,yAxisID:'y'},
      {label:'Avg Frequency (x)',data:prof.map(p=>p.Avg_Frequency),backgroundColor:'#5b8def',borderRadius:8,yAxisID:'y'},
      {label:'Avg Monetary',data:prof.map(p=>p.Avg_Monetary),backgroundColor:'#e0a458',borderRadius:8,yAxisID:'y1'}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{usePointStyle:true}}},
        scales:{x:{grid:{display:false}},y:{position:'left',title:{display:true,text:'Recency / Frequency'},grid:{color:'rgba(15,118,110,.08)'}},
          y1:{position:'right',title:{display:true,text:'Monetary'},grid:{display:false},ticks:{callback:v=>v>=1e6?(v/1e6).toFixed(1)+' jt':v}}}}});
    const sc={{ kmeans.scatter|tojson }};
    new Chart(document.getElementById('chart_scatter'),{type:'scatter',data:{datasets:Object.keys(COL).map(l=>({label:l,
      data:sc.filter(d=>d.Label_Bisnis===l).map(d=>({x:d.Frequency,y:d.Monetary})),backgroundColor:COL[l]+'b3',pointRadius:5,pointHoverRadius:8}))},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{usePointStyle:true}}},
        scales:{x:{title:{display:true,text:'Frequency'},grid:{color:'rgba(15,118,110,.08)'}},y:{title:{display:true,text:'Monetary'},grid:{color:'rgba(15,118,110,.08)'},ticks:{callback:v=>v>=1e6?(v/1e6).toFixed(1)+' jt':v}}}}});
  });
  </script>
  {% endif %}

{% elif active_view == 'kmanual' %}
  {% if not prep_ok %}{{ gate('Lakukan Transformasi & Standardisasi terlebih dahulu.', 'prep', 'Ke Transformasi') }}
  {% else %}
  <!-- FORM INPUT K MANUAL -->
  <div class="max-w-2xl mx-auto glass rounded-3xl p-8 text-center shadow-xl mb-5">
    <div class="w-20 h-20 mx-auto rounded-3xl flex items-center justify-center mb-4 shadow-xl" style="background:linear-gradient(135deg,#0d9488,#0e7490)"><i class="fas fa-sliders text-3xl text-white"></i></div>
    <h3 class="text-2xl font-extrabold mb-2">K Manual</h3>
    <p class="text-sm mb-5" style="color:var(--ink2)">Tentukan sendiri jumlah cluster (k), lalu K-Means langsung dijalankan pada data hasil transformasi. Setiap cluster ditampilkan dengan nomor dan keterangan teks agar hasilnya mudah dianalisis.</p>
    <form action="/kmanual" method="POST" class="flex flex-wrap items-end justify-center gap-4">
      <div class="text-left">
        <label for="kInput" class="block text-[11px] font-bold mb-1" style="color:var(--muted)">Jumlah cluster (k) · 2 sampai {{ kmanual_max }}</label>
        <input id="kInput" class="kinp" type="number" name="k" min="2" max="{{ kmanual_max }}" step="1" value="{{ kmanual_default }}" required>
      </div>
      <button class="btn" type="submit"><i class="fas fa-play"></i> Jalankan K-Means</button>
    </form>
    {% if kmanual.error %}<p class="text-xs mt-4 font-semibold" style="color:#a22c40">{{ kmanual.error }}</p>{% endif %}
  </div>

  {% if kmanual_ok %}
  {% set kpal = ['#0d9488','#5b8def','#e36f7f','#e0a458','#8b5cf6','#0e7490','#64748b','#f97316','#84cc16','#ec4899'] %}
  <div class="hero">
    <div class="flex items-start justify-between flex-wrap gap-4 mb-5">
      <div><span class="inline-flex items-center gap-2 bg-white/20 px-3 py-1.5 rounded-full text-xs font-bold mb-3"><i class="fas fa-sliders"></i> Bagian 4B · K Manual</span>
        <h1 class="text-3xl font-extrabold mb-1">Hasil K-Means (k = {{ kmanual.k }})</h1>
        <p class="text-sm opacity-90">Cluster dinyatakan dengan nomor dan keterangan teks. Konvergen dalam {{ kmanual.iterations }} iterasi.</p></div>
      <div class="text-right"><p class="text-xs opacity-80 font-bold">Jumlah cluster</p><p class="text-5xl font-extrabold">{{ kmanual.k }}</p></div>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Total pelanggan</p><p class="text-2xl font-extrabold">{{ fi(kmanual.labeled_df|length) }}</p></div>
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Inertia / WCSS</p><p class="text-2xl font-extrabold mono">{{ '%.2f'|format(kmanual.inertia) }}</p></div>
      <div class="kpi"><p class="text-[11px] font-bold opacity-80">Iterasi sampai konvergen</p><p class="text-2xl font-extrabold">{{ kmanual.iterations }}</p></div>
    </div>
  </div>

  <div class="sec">
    {{ sech('fa-table', 'linear-gradient(135deg,#0d9488,#0e7490)', 'Profil cluster', 'Nomor cluster, keterangan, dan rata-rata Recency, Frequency, Monetary') }}
    {{ profile_table_manual(kmanual.profile) }}
    <div class="ch mt-5"><canvas id="km_profile"></canvas></div>
  </div>

  <div class="sec">
    {{ sech('fa-list-check', 'linear-gradient(135deg,#c98a3e,#e0a458)', 'Keterangan tiap cluster', 'Interpretasi otomatis dari profil RFM, diurutkan dari nilai pelanggan tertinggi') }}
    {% for r in kmanual.profile.sort_values('Peringkat').to_dict('records') %}
      {% set cidx = r.Cluster|int %}
      <div class="strat" style="border-color:{{ kpal[cidx % 10] }};background:linear-gradient(135deg,{{ kpal[cidx % 10] }}1a,#fff)">
        <div class="flex items-center flex-wrap gap-3 mb-2">
          <span class="cnum" style="background:{{ kpal[cidx % 10] }}">{{ cidx }}</span>
          <span class="text-sm font-extrabold" style="color:var(--ink)">{{ r.Keterangan }}</span>
          <span class="text-xs font-semibold" style="color:var(--muted)">{{ fi(r.Jumlah_Pelanggan) }} pelanggan · Peringkat nilai #{{ r.Peringkat }} dari {{ kmanual.k }}</span>
        </div>
        <div class="flex flex-wrap gap-2 mb-2">
          <span class="text-[11px] font-bold" style="color:var(--muted)">Recency</span><span class="lv lv-{{ r.Level_R }}">{{ r.Level_R }}</span>
          <span class="text-[11px] font-bold ml-2" style="color:var(--muted)">Frequency</span><span class="lv lv-{{ r.Level_F }}">{{ r.Level_F }}</span>
          <span class="text-[11px] font-bold ml-2" style="color:var(--muted)">Monetary</span><span class="lv lv-{{ r.Level_M }}">{{ r.Level_M }}</span>
        </div>
        <p class="text-xs mb-1" style="color:var(--ink)"><b>Karakteristik:</b> {{ r.Karakteristik }}</p>
        <p class="text-xs" style="color:var(--ink2)"><b>Saran tindakan:</b> {{ r.Saran }}</p>
      </div>
    {% endfor %}
    <div class="qa" style="border-left-color:var(--teal2);margin-top:8px">
      <p class="text-xs font-extrabold mb-1" style="color:var(--teal)"><i class="fas fa-circle-info mr-1"></i> Cara membaca keterangan</p>
      <p class="text-xs leading-relaxed" style="color:var(--ink2)">Tingkat Rendah / Sedang / Tinggi ditentukan dari rata-rata z-score tiap cluster pada hasil standardisasi: <b>Tinggi</b> bila z &ge; 0,5, <b>Rendah</b> bila z &le; -0,5, selain itu <b>Sedang</b>. Khusus Recency, nilai <b>Rendah</b> berarti pelanggan baru-baru ini bertransaksi (lebih baik). Peringkat nilai dihitung dari kombinasi Recency rendah, Frequency tinggi, dan Monetary tinggi. Keterangan bersifat interpretasi otomatis dan dapat berubah ketika k diganti.</p>
    </div>
  </div>

  <div class="sec">
    {{ sech('fa-bullhorn', 'linear-gradient(135deg,#c98a3e,#e0a458)', 'Strategi Marketing', 'Tujuan, aksi, kanal, KPI, dan prioritas anggaran untuk tiap cluster (urut dari nilai pelanggan tertinggi)') }}
    <div class="tbl-wrap mb-5"><table class="t"><thead><tr><th>Cluster</th><th>Keterangan</th><th>Jumlah Pelanggan</th><th>Prioritas Anggaran</th></tr></thead><tbody>
      {% for r in kmanual.profile.sort_values('Peringkat').to_dict('records') %}{% set cidx = r.Cluster|int %}
      <tr><td><span class="cnum" style="background:{{ kpal[cidx % 10] }}">{{ cidx }}</span></td><td class="font-bold">{{ r.Keterangan }}</td><td class="mono">{{ fi(r.Jumlah_Pelanggan) }}</td><td><span class="chip chip-t">{{ r.Anggaran }}</span></td></tr>
      {% endfor %}
    </tbody></table></div>
    {% for r in kmanual.profile.sort_values('Peringkat').to_dict('records') %}
      {% set cidx = r.Cluster|int %}
      <div class="strat" style="border-color:{{ kpal[cidx % 10] }};background:linear-gradient(135deg,{{ kpal[cidx % 10] }}1a,#fff)">
        <div class="flex items-center flex-wrap gap-3 mb-2">
          <span class="cnum" style="background:{{ kpal[cidx % 10] }}">{{ cidx }}</span>
          <span class="text-sm font-extrabold" style="color:var(--ink)">{{ r.Keterangan }}</span>
          <span class="chip chip-a"><i class="fas fa-coins"></i> Anggaran: {{ r.Anggaran }}</span>
        </div>
        <p class="text-xs mb-2" style="color:var(--ink)"><b>Tujuan:</b> {{ r.Tujuan }}</p>
        <p class="text-xs font-bold mb-1" style="color:var(--ink)">Aksi yang disarankan:</p>
        <ul class="text-xs ml-5 list-disc space-y-1 mb-2" style="color:var(--ink2)">{% for a in r.Aksi.split('||') %}<li>{{ a }}</li>{% endfor %}</ul>
        <p class="text-xs mb-1" style="color:var(--ink2)"><b>Kanal:</b> {{ r.Kanal }}</p>
        <p class="text-xs" style="color:var(--ink2)"><b>KPI keberhasilan:</b> {{ r.KPI }}</p>
      </div>
    {% endfor %}
    <div class="rounded-2xl p-5 mt-4 text-white" style="background:linear-gradient(135deg,#0f4a4f,#0f766e)">
      <h4 class="font-extrabold mb-2"><i class="fas fa-coins mr-2"></i>Mengurangi pemborosan anggaran pemasaran</h4>
      <ul class="text-xs space-y-2 ml-5 list-disc opacity-95">
        <li>Alihkan budget iklan massal ke kampanye tersegmentasi sesuai keterangan tiap cluster.</li>
        <li>Prioritaskan anggaran pada cluster bernilai tinggi dan potensial, karena dampak revenue-nya paling besar.</li>
        <li>Batasi biaya pada cluster bernilai rendah dan pasif; beri penawaran win-back hanya dalam periode terbatas.</li>
        <li>Ukur KPI tiap cluster secara berkala, lalu realokasi budget ke cluster dengan hasil terbaik.</li>
        <li>Jika k diganti, strategi ikut menyesuaikan dengan keterangan baru.</li>
      </ul>
    </div>
  </div>

  <div class="sec">
    {{ sech('fa-braille', 'linear-gradient(135deg,#5b8def,#85abf5)', 'Visualisasi cluster', 'Frequency vs Monetary (sampel hingga 800 pelanggan)') }}
    <div class="ch-tall"><canvas id="km_scatter"></canvas></div>
  </div>

  <div class="sec">
    {{ sech('fa-users', 'linear-gradient(135deg,#c98a3e,#e0a458)', 'Pelanggan + nomor cluster + keterangan', '50 baris pertama') }}
    {{ table(kmanual.table) }}
    <p class="text-[11px] mt-2 text-center" style="color:var(--muted)"><i class="fas fa-circle-info mr-1"></i>Menampilkan 50 dari {{ fi(kmanual.labeled_df|length) }} pelanggan. Untuk melihat semua data, gunakan tombol unduh.</p>
    <div class="mt-3 text-center"><a href="/download/kmanual" class="btn"><i class="fas fa-download"></i> Unduh pelanggan + cluster + keterangan (CSV)</a></div>
  </div>

  <script>
  const KM_PAL=['#0d9488','#5b8def','#e36f7f','#e0a458','#8b5cf6','#0e7490','#64748b','#f97316','#84cc16','#ec4899'];
  const kmShort=v=>v>=1e9?(v/1e9).toFixed(1)+' M':v>=1e6?(v/1e6).toFixed(1)+' jt':v>=1e3?(v/1e3).toFixed(0)+' rb':v;
  document.addEventListener('DOMContentLoaded',()=>{
    const prof={{ kmanual.profile_json|tojson }};
    new Chart(document.getElementById('km_profile'),{type:'bar',data:{labels:prof.map(p=>'C'+p.Cluster+' · '+p.Keterangan),datasets:[
      {label:'Avg Recency (hari)',data:prof.map(p=>p.Avg_Recency),backgroundColor:'#0e7490',borderRadius:8,yAxisID:'y'},
      {label:'Avg Frequency (x)',data:prof.map(p=>p.Avg_Frequency),backgroundColor:'#5b8def',borderRadius:8,yAxisID:'y'},
      {label:'Avg Monetary',data:prof.map(p=>p.Avg_Monetary),backgroundColor:'#e0a458',borderRadius:8,yAxisID:'y1'}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{usePointStyle:true}}},
        scales:{x:{grid:{display:false}},y:{position:'left',title:{display:true,text:'Recency / Frequency'},grid:{color:'rgba(15,118,110,.08)'}},
          y1:{position:'right',title:{display:true,text:'Monetary'},grid:{display:false},ticks:{callback:kmShort}}}}});
    const sc={{ kmanual.scatter|tojson }};
    new Chart(document.getElementById('km_scatter'),{type:'scatter',data:{datasets:prof.map((p,i)=>({label:'Cluster '+p.Cluster+' · '+p.Keterangan,
      data:sc.filter(d=>d.Cluster===p.Cluster).map(d=>({x:d.Frequency,y:d.Monetary})),backgroundColor:KM_PAL[i%KM_PAL.length]+'b3',pointRadius:5,pointHoverRadius:8}))},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{usePointStyle:true}}},
        scales:{x:{title:{display:true,text:'Frequency'},grid:{color:'rgba(15,118,110,.08)'}},y:{title:{display:true,text:'Monetary'},grid:{color:'rgba(15,118,110,.08)'},ticks:{callback:kmShort}}}}});
  });
  </script>
  {% endif %}
  {% endif %}

{% elif active_view == 'dashboard' %}
  {% if not kmeans_ok %}{{ gate('Lakukan K-Means terlebih dahulu.', 'kmeans', 'Ke K-Means') }}
  {% elif not dash_ok %}{{ runcard('fa-chart-pie', 'Dashboard BI', 'Tampilkan KPI, komposisi segmen, revenue per segmen, tren bulanan, scatter, profil cluster, dan strategi.', '/dashboard', 'Buka Dashboard') }}
  {% else %}
  <div class="mb-4">
    <h1 class="text-2xl font-extrabold">Dashboard Segmentasi Pelanggan</h1>
    <p class="text-sm" style="color:var(--ink2)">Ringkasan perilaku pelanggan TokoKita berdasarkan RFM dan K-Means (acuan 30 Juni 2024).</p>
  </div>

  <!-- KPI CARD -->
  <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 mb-5">
    <div class="kpi-card"><div class="kpi-ic" style="background:linear-gradient(135deg,#0d9488,#2aa89b)"><i class="fas fa-users"></i></div><div><p class="text-[11px] font-bold" style="color:var(--muted)">Total Pelanggan</p><p class="text-2xl font-extrabold">{{ fi(dash.kpi.customers) }}</p></div></div>
    <div class="kpi-card"><div class="kpi-ic" style="background:linear-gradient(135deg,#0e7490,#22a1bd)"><i class="fas fa-receipt"></i></div><div><p class="text-[11px] font-bold" style="color:var(--muted)">Total Transaksi</p><p class="text-2xl font-extrabold">{{ fi(dash.kpi.transactions) }}</p></div></div>
    <div class="kpi-card"><div class="kpi-ic" style="background:linear-gradient(135deg,#5b8def,#85abf5)"><i class="fas fa-sack-dollar"></i></div><div><p class="text-[11px] font-bold" style="color:var(--muted)">Total Revenue</p><p class="text-lg font-extrabold">{{ rp(dash.kpi.revenue) }}</p></div></div>
    <div class="kpi-card"><div class="kpi-ic" style="background:linear-gradient(135deg,#c98a3e,#e0a458)"><i class="fas fa-scale-balanced"></i></div><div><p class="text-[11px] font-bold" style="color:var(--muted)">Average Monetary</p><p class="text-lg font-extrabold">{{ rp(dash.kpi.avg_monetary) }}</p></div></div>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
    <div class="sec">{{ sech('fa-chart-pie', 'linear-gradient(135deg,#0d9488,#0e7490)', 'Komposisi Segmen', 'Proporsi Loyal / New / Churn Risk (donut)') }}<div class="ch"><canvas id="c_pie"></canvas></div></div>
    <div class="sec">{{ sech('fa-chart-column', 'linear-gradient(135deg,#5b8def,#85abf5)', 'Revenue per Segmen', 'Kontribusi tiap segmen (column chart)') }}<div class="ch"><canvas id="c_rev"></canvas></div></div>
    <div class="sec">{{ sech('fa-chart-line', 'linear-gradient(135deg,#0e7490,#22a1bd)', 'Tren Bulanan', 'Transaksi dan revenue per bulan (line chart)') }}<div class="ch"><canvas id="c_month"></canvas></div></div>
    <div class="sec">{{ sech('fa-braille', 'linear-gradient(135deg,#c98a3e,#e0a458)', 'Frequency vs Monetary', 'Pola pelanggan (scatter plot)') }}<div class="ch"><canvas id="c_sc"></canvas></div></div>
  </div>

  <div class="sec">
    {{ sech('fa-table', 'linear-gradient(135deg,#0d9488,#2aa89b)', 'Profil Cluster', 'Perbandingan RFM tiap cluster (tabel dan bar)') }}
    {{ profile_table(kmeans.profile, True) }}
    <div class="ch mt-5"><canvas id="c_prof"></canvas></div>
  </div>

  <div class="sec">
    {{ sech('fa-bullhorn', 'linear-gradient(135deg,#c98a3e,#e0a458)', 'Strategi Marketing', 'Aksi berdasarkan segmen (insight cards)') }}
    {% for r in kmeans.profile.to_dict('records') %}
      <div class="strat {{ 'strat-Loyal' if r.Label_Bisnis == 'Loyal' else ('strat-New' if r.Label_Bisnis == 'New' else 'strat-Churn') }}">
        <div class="flex items-center gap-3 mb-2">{{ segbadge(r.Label_Bisnis) }}<span class="text-xs font-semibold" style="color:var(--muted)">{{ fi(r.Jumlah_Pelanggan) }} pelanggan</span></div>
        <p class="text-xs mb-1" style="color:var(--ink)"><b>Karakteristik:</b> {{ r.Karakteristik }}</p>
        <p class="text-xs" style="color:var(--ink2)"><b>Strategi:</b> {{ r.Strategi }}</p></div>
    {% endfor %}
    {{ qa('dashboard') }}
  </div>

  <script>
  const COL={'Loyal':'#0d9488','New':'#5b8def','Churn Risk':'#e36f7f'};
  const short=v=>v>=1e9?(v/1e9).toFixed(1)+' M':v>=1e6?(v/1e6).toFixed(1)+' jt':v>=1e3?(v/1e3).toFixed(0)+' rb':v;
  const leg={position:'bottom',labels:{usePointStyle:true,padding:14}};
  document.addEventListener('DOMContentLoaded',()=>{
    const comp={{ dash.composition|tojson }}, mon={{ dash.monthly|tojson }}, sc={{ dash.scatter|tojson }}, prof={{ dash.profile|tojson }};
    new Chart(c_pie,{type:'doughnut',data:{labels:comp.map(c=>c.Label_Bisnis),datasets:[{data:comp.map(c=>c.Jumlah),backgroundColor:comp.map(c=>COL[c.Label_Bisnis]),borderWidth:5,borderColor:'#fff',hoverOffset:14}]},
      options:{responsive:true,maintainAspectRatio:false,cutout:'58%',plugins:{legend:leg,tooltip:{callbacks:{label:x=>{const t=x.dataset.data.reduce((a,b)=>a+b,0);return x.label+': '+x.parsed+' ('+(x.parsed/t*100).toFixed(1)+'%)'}}}}}});
    new Chart(c_rev,{type:'bar',data:{labels:comp.map(c=>c.Label_Bisnis),datasets:[{data:comp.map(c=>c.Revenue),backgroundColor:comp.map(c=>COL[c.Label_Bisnis]),borderRadius:10}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:x=>'Revenue: Rp '+x.parsed.y.toLocaleString('id-ID')}}},scales:{x:{grid:{display:false}},y:{grid:{color:'rgba(15,118,110,.08)'},ticks:{callback:short}}}}});
    new Chart(c_month,{type:'line',data:{labels:mon.map(m=>m.YearMonth),datasets:[
      {label:'Transaksi',data:mon.map(m=>m.Transaksi),borderColor:'#0d9488',backgroundColor:'rgba(13,148,136,.14)',borderWidth:3,fill:true,tension:.3,yAxisID:'y'},
      {label:'Revenue',data:mon.map(m=>m.Revenue),borderColor:'#5b8def',backgroundColor:'rgba(91,141,239,.10)',borderWidth:3,fill:false,tension:.3,yAxisID:'y1'}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:leg},scales:{x:{grid:{display:false}},y:{position:'left',grid:{color:'rgba(15,118,110,.08)'}},y1:{position:'right',grid:{display:false},ticks:{callback:short}}}}});
    new Chart(c_sc,{type:'scatter',data:{datasets:Object.keys(COL).map(l=>({label:l,data:sc.filter(d=>d.Label_Bisnis===l).map(d=>({x:d.Frequency,y:d.Monetary})),backgroundColor:COL[l]+'b3',pointRadius:5}))},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:leg},scales:{x:{title:{display:true,text:'Frequency'},grid:{color:'rgba(15,118,110,.08)'}},y:{title:{display:true,text:'Monetary'},grid:{color:'rgba(15,118,110,.08)'},ticks:{callback:short}}}}});
    new Chart(c_prof,{type:'bar',data:{labels:prof.map(p=>p.Label_Bisnis+' (C'+p.Cluster+')'),datasets:[
      {label:'Avg Recency (hari)',data:prof.map(p=>p.Avg_Recency),backgroundColor:'#0e7490',borderRadius:8},
      {label:'Avg Frequency (x)',data:prof.map(p=>p.Avg_Frequency),backgroundColor:'#5b8def',borderRadius:8},
      {label:'Avg Monetary',data:prof.map(p=>p.Avg_Monetary),backgroundColor:'#e0a458',borderRadius:8,yAxisID:'y1'}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:leg},scales:{x:{grid:{display:false}},y:{position:'left',grid:{color:'rgba(15,118,110,.08)'}},y1:{position:'right',grid:{display:false},ticks:{callback:short}}}}});
  });
  </script>
  {% endif %}

{% elif active_view == 'strategy' %}
  {% if not kmeans_ok %}{{ gate('Lakukan K-Means terlebih dahulu.', 'kmeans', 'Ke K-Means') }}
  {% else %}
  <div class="sec">
    {{ sech('fa-bullhorn', 'linear-gradient(135deg,#c98a3e,#e0a458)', 'Strategi marketing per segmen', 'Loyal, New, dan Churn Risk') }}
    {% for r in kmeans.profile.to_dict('records') %}
      <div class="strat {{ 'strat-Loyal' if r.Label_Bisnis == 'Loyal' else ('strat-New' if r.Label_Bisnis == 'New' else 'strat-Churn') }}">
        <div class="flex items-center gap-3 mb-2">{{ segbadge(r.Label_Bisnis) }}<span class="text-xs font-semibold" style="color:var(--muted)">{{ fi(r.Jumlah_Pelanggan) }} pelanggan</span></div>
        <p class="text-xs mb-1"><b>Karakteristik:</b> {{ r.Karakteristik }}</p><p class="text-xs" style="color:var(--ink2)"><b>Strategi:</b> {{ r.Strategi }}</p></div>
    {% endfor %}
    <div class="rounded-2xl p-5 mt-4 text-white" style="background:linear-gradient(135deg,#0f4a4f,#0f766e)">
      <h4 class="font-extrabold mb-2"><i class="fas fa-coins mr-2"></i>Mengurangi pemborosan anggaran pemasaran</h4>
      <ul class="text-xs space-y-2 ml-5 list-disc opacity-95">
        <li>Alihkan budget iklan massal ke kampanye tersegmentasi: retensi (Loyal), onboarding (New), win-back (Churn Risk).</li>
        <li>Hentikan diskon besar untuk Loyal, cukup beri reward eksklusif bernilai tinggi.</li>
        <li>Fokuskan biaya akuisisi lanjutan pada New dengan penawaran yang ROI-nya terukur.</li>
        <li>Beri penawaran win-back hemat biaya pada Churn Risk sebelum pelanggan benar-benar hilang.</li>
        <li>Ukur ROI per segmen dan realokasi budget ke segmen berkontribusi revenue tertinggi.</li>
      </ul>
    </div>
  </div>
  {% endif %}

{% elif active_view == 'conclusion' %}
  {% if not kmeans_ok %}{{ gate('Lakukan K-Means terlebih dahulu.', 'kmeans', 'Ke K-Means') }}
  {% else %}
  <div class="sec">
    {{ sech('fa-clipboard-check', 'linear-gradient(135deg,#0d9488,#0e7490)', 'Kesimpulan akhir', 'Ringkasan seluruh alur analisis') }}
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
      <div class="stat" style="background:linear-gradient(135deg,#0d9488,#2aa89b)"><p class="text-[11px] font-bold opacity-90">Total pelanggan</p><p class="text-3xl font-extrabold">{{ fi(rfm.n_customers) }}</p></div>
      <div class="stat" style="background:linear-gradient(135deg,#0e7490,#22a1bd)"><p class="text-[11px] font-bold opacity-90">Total revenue</p><p class="text-xl font-extrabold">{{ rp(rfm.rfm_df['Monetary'].sum()) }}</p></div>
      <div class="stat" style="background:linear-gradient(135deg,#5b8def,#85abf5)"><p class="text-[11px] font-bold opacity-90">Jumlah cluster</p><p class="text-3xl font-extrabold">{{ kmeans.k }}</p></div>
    </div>
    <div class="rounded-2xl p-6 text-white" style="background:linear-gradient(135deg,#0f4a4f,#0f766e)">
      <ol class="text-xs space-y-2 list-decimal ml-5 opacity-95">
        <li>Data mentah dibersihkan dari <b>{{ fi(clean.rows_before) }}</b> menjadi <b>{{ fi(clean.rows_after) }}</b> baris (duplikat persis {{ fi(clean.n_dup_exact) }}, duplikat transaksi {{ fi(clean.n_dup_trx) }}, nilai kosong {{ fi(clean.n_missing_before) }}).</li>
        <li>RFM dihitung untuk <b>{{ fi(rfm.n_customers) }}</b> pelanggan dengan Recency acuan 30 Juni 2024.</li>
        <li>Frequency dan Monetary ditransformasi LN(x+1) lalu ketiga fitur distandardisasi.</li>
        <li>Elbow Method menyarankan k = <b>{{ elbow.optimal_k }}</b>; K-Means konvergen dalam {{ kmeans.iterations }} iterasi.</li>
        <li>Pelanggan dibagi menjadi segmen {% for c in kmeans.composition.to_dict('records') %}{{ c.Label_Bisnis }} ({{ fi(c.Jumlah) }}){{ ', ' if not loop.last else '' }}{% endfor %}.</li>
        <li>Strategi marketing disusun per segmen untuk menekan pemborosan anggaran iklan massal.</li>
      </ol>
    </div>
  </div>
  {% endif %}
{% endif %}
  </div>
</main>
</div>

<script>
const fi_=document.getElementById('fileInput'), fn=document.getElementById('fileName'), sb=document.getElementById('submitBtn');
if(fi_) fi_.addEventListener('change',e=>{const f=e.target.files[0]; if(f) fn.innerHTML='<i class="fas fa-circle-check mr-1" style="color:#0d9488"></i><b>'+f.name+'</b> · '+(f.size/1024).toFixed(1)+' KB';});
if(sb) sb.closest('form').addEventListener('submit',()=>{sb.innerHTML='<i class="fas fa-circle-notch animate-spin"></i> Memproses data...';sb.disabled=true;});
</script>
</body>
</html>
"""

# Macro tabel profil cluster dipasang sebagai fungsi Python agar template tetap ringkas
def profile_table(prof_df, with_log):
    from markupsafe import Markup
    head = ['Cluster', 'Jumlah Pelanggan', 'Avg Recency', 'Avg Frequency', 'Avg Monetary']
    if with_log:
        head += ['Avg Log Frequency', 'Avg Log Monetary']
    head += ['Label Bisnis']
    h = ''.join(f'<th>{x}</th>' for x in head)
    rows = ''
    badge = {
        'Loyal': '<span class="seg seg-Loyal"><i class="fas fa-crown"></i> Loyal</span>',
        'New': '<span class="seg seg-New"><i class="fas fa-seedling"></i> New</span>',
        'Churn Risk': '<span class="seg seg-Churn"><i class="fas fa-triangle-exclamation"></i> Churn Risk</span>',
    }
    for r in prof_df.to_dict('records'):
        cells = [f"<b>{int(r['Cluster'])}</b>", format_int(r['Jumlah_Pelanggan']),
                 f"{r['Avg_Recency']:.1f} hari", f"{r['Avg_Frequency']:.2f}", format_rupiah(r['Avg_Monetary'])]
        if with_log:
            cells += [f"{r['Avg_Log_Frequency']:.4f}", f"{r['Avg_Log_Monetary']:.4f}"]
        cells.append(badge.get(r['Label_Bisnis'], r['Label_Bisnis']))
        rows += '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
    return Markup(f'<div class="tbl-wrap"><table class="t"><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table></div>')


# Tabel profil cluster untuk K Manual: nomor cluster + keterangan teks
def profile_table_manual(prof_df):
    from markupsafe import Markup, escape
    head = ['Cluster', 'Keterangan', 'Peringkat Nilai', 'Jumlah Pelanggan', 'Avg Recency', 'Avg Frequency',
            'Avg Monetary', 'Avg Log Frequency', 'Avg Log Monetary']
    h = ''.join(f'<th>{x}</th>' for x in head)
    pal = ['#0d9488', '#5b8def', '#e36f7f', '#e0a458', '#8b5cf6', '#0e7490', '#64748b', '#f97316', '#84cc16', '#ec4899']
    rows = ''
    for r in prof_df.to_dict('records'):
        color = pal[int(r['Cluster']) % len(pal)]
        num = f'<span class="cnum" style="background:{color}">{int(r["Cluster"])}</span>'
        ket = f'<span class="ket" style="background:{color}">{escape(r["Keterangan"])}</span>'
        cells = [num, ket, f"#{int(r['Peringkat'])}", format_int(r['Jumlah_Pelanggan']),
                 f"{r['Avg_Recency']:.1f} hari", f"{r['Avg_Frequency']:.2f}", format_rupiah(r['Avg_Monetary']),
                 f"{r['Avg_Log_Frequency']:.4f}", f"{r['Avg_Log_Monetary']:.4f}"]
        rows += '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
    return Markup(f'<div class="tbl-wrap"><table class="t"><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table></div>')


# ============================================================
# ROUTES
# ============================================================
PAGES = {
    'upload': ('Unggah Data', 'fa-file-arrow-up'),
    'sheets': ('Preview Data Mentah', 'fa-table'),
    'cleaning': ('Data Cleaning', 'fa-broom'),
    'rfm': ('Perhitungan RFM', 'fa-chart-simple'),
    'prep': ('Transformasi & Standardisasi', 'fa-wand-magic-sparkles'),
    'elbow': ('Elbow Method', 'fa-chart-line'),
    'kmeans': ('K-Means & Interpretasi', 'fa-object-group'),
    'kmanual': ('K Manual', 'fa-sliders'),
    'dashboard': ('Dashboard BI', 'fa-chart-pie'),
    'strategy': ('Strategi Marketing', 'fa-bullhorn'),
    'conclusion': ('Kesimpulan', 'fa-clipboard-check'),
}


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '').strip()
        if u == USERNAME_VALID and p == PASSWORD_VALID:
            session['logged_in'] = True
            session['username'] = u
            flash(f'Selamat datang, {u.capitalize()}!', 'success')
            return redirect(url_for('index'))
        flash('Username atau password salah. Coba lagi.', 'error')
        return redirect(url_for('login'))
    return render_template_string(LOGIN_TEMPLATE)


@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah keluar.', 'success')
    return redirect(url_for('login'))


@app.route('/')
@login_required
def index():
    sheets = list(UPLOADED_DATA.keys())
    view = request.args.get('view', 'sheets' if sheets else 'upload')
    if view not in PAGES:
        view = 'sheets' if sheets else 'upload'

    # Dashboard dihitung otomatis saat K-Means sudah selesai
    if view == 'dashboard' and KMEANS_DATA.get('success') and not DASHBOARD_DATA.get('success'):
        run_dashboard()

    cok = CLEANING_DATA.get('success', False)
    rok = RFM_DATA.get('success', False)
    pok = PREP_DATA.get('success', False)
    eok = ELBOW_DATA.get('success', False)
    kok = KMEANS_DATA.get('success', False)
    kmok = KMANUAL_DATA.get('success', False)
    dok = DASHBOARD_DATA.get('success', False)

    nav_main = [
        ('sheets', '1. Preview Data Mentah', 'fa-table', bool(sheets)),
        ('cleaning', '2. Data Cleaning', 'fa-broom', cok),
        ('rfm', '3. Perhitungan RFM', 'fa-chart-simple', rok),
        ('prep', '4. Transformasi', 'fa-wand-magic-sparkles', pok),
        ('elbow', '5. Elbow Method', 'fa-chart-line', eok),
        ('kmeans', '6. K-Means', 'fa-object-group', kok),
        ('kmanual', '7. K Manual', 'fa-sliders', kmok),
    ]
    nav_out = [
        ('dashboard', 'Dashboard BI', 'fa-chart-pie', dok),
        ('strategy', 'Strategi Marketing', 'fa-bullhorn', False),
        ('conclusion', 'Kesimpulan', 'fa-clipboard-check', False),
    ]

    raw_rows = raw_cols = 0
    raw_table = ([], [])
    if sheets:
        raw_df = UPLOADED_DATA[sheets[0]]
        raw_rows, raw_cols = format_int(len(raw_df)), len(raw_df.columns)
        raw_table = df_to_table(raw_df, 100)

    pct_clean = 0
    if cok and CLEANING_DATA.get('rows_before'):
        pct_clean = round(CLEANING_DATA['rows_after'] / CLEANING_DATA['rows_before'] * 100, 1)

    rfm_tpl = ([], [])
    if view == 'rfm' and rok:
        rfm_tpl = rfm_template_display(build_rfm_template(), 50)

    # batas & nilai awal input K Manual
    kmanual_max = 10
    if pok:
        kmanual_max = max(2, min(10, len(PREP_DATA['prep_df']) - 1))
    kmanual_default = KMANUAL_DATA.get('k') or ELBOW_DATA.get('optimal_k') or 3
    kmanual_default = max(2, min(int(kmanual_default), kmanual_max))

    title, icon = PAGES[view]
    return render_template_string(
        BASE_TEMPLATE,
        rfm_tpl=rfm_tpl,
        sheets=sheets, active_view=view, page_title=title, page_icon=icon,
        nav_main=nav_main, nav_out=nav_out,
        raw_rows=raw_rows, raw_cols=raw_cols, raw_table=raw_table,
        clean=CLEANING_DATA, rfm=RFM_DATA, prep=PREP_DATA, elbow=ELBOW_DATA,
        kmeans=KMEANS_DATA, kmanual=KMANUAL_DATA, dash=DASHBOARD_DATA,
        cleaning_ok=cok, rfm_ok=rok, prep_ok=pok, elbow_ok=eok, kmeans_ok=kok, kmanual_ok=kmok, dash_ok=dok,
        kmanual_max=kmanual_max, kmanual_default=kmanual_default,
        pct_clean=pct_clean, answers=ANSWERS,
        fi=format_int, rp=format_rupiah, profile_table=profile_table,
        profile_table_manual=profile_table_manual,
    )


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    global UPLOADED_DATA, CLEANING_DATA, RFM_DATA, PREP_DATA, ELBOW_DATA, KMEANS_DATA, KMANUAL_DATA, DASHBOARD_DATA
    if 'file' not in request.files or request.files['file'].filename == '':
        flash('Tidak ada file yang dipilih.', 'error')
        return redirect(url_for('index', view='upload'))
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    try:
        try:
            xls = pd.ExcelFile(filepath, engine='calamine')
        except Exception:
            xls = pd.ExcelFile(filepath)
        raw_name = find_raw_sheet_name(xls.sheet_names)
        if raw_name is None:
            flash("Sheet 'Data Transaksi Mentah' tidak ditemukan. Sheet di file: " + ', '.join(xls.sheet_names), 'error')
            return redirect(url_for('index', view='upload'))
        df = pd.read_excel(xls, sheet_name=raw_name, dtype=object)
        df.columns = df.columns.astype(str).str.strip()
        df = df.dropna(how='all').reset_index(drop=True)

        UPLOADED_DATA = {raw_name: df}      # hanya sheet mentah yang disimpan
        CLEANING_DATA, RFM_DATA, PREP_DATA = {}, {}, {}
        ELBOW_DATA, KMEANS_DATA, KMANUAL_DATA, DASHBOARD_DATA = {}, {}, {}, {}
        flash(f"Sheet '{raw_name}' berhasil dimuat ({len(df)} baris). Sheet lain diabaikan.", 'success')
        return redirect(url_for('index', view='sheets'))
    except Exception as e:
        flash(f'Gagal membaca file: {e}', 'error')
        return redirect(url_for('index', view='upload'))


def _run_step(fn, ok_msg, view, reset, *args):
    """Jalankan satu langkah, reset hasil langkah berikutnya, lalu arahkan ke halaman."""
    for name in reset:
        globals()[name] = {}
    result = fn(*args)
    if result.get('success'):
        flash(ok_msg(result), 'success')
    else:
        flash(f"Gagal: {result.get('error')}", 'error')
    return redirect(url_for('index', view=view))


@app.route('/cleaning', methods=['POST'])
@login_required
def cleaning():
    if not UPLOADED_DATA:
        flash('Belum ada data.', 'error')
        return redirect(url_for('index', view='upload'))
    return _run_step(run_data_cleaning, lambda r: f"Data Cleaning selesai: {r['rows_after']} baris bersih.",
                     'cleaning', ['RFM_DATA', 'PREP_DATA', 'ELBOW_DATA', 'KMEANS_DATA', 'KMANUAL_DATA', 'DASHBOARD_DATA'])


@app.route('/rfm', methods=['POST'])
@login_required
def rfm():
    if not CLEANING_DATA.get('success'):
        flash('Lakukan Data Cleaning terlebih dahulu.', 'error')
        return redirect(url_for('index', view='cleaning'))
    return _run_step(run_rfm, lambda r: f"RFM selesai untuk {r['n_customers']} pelanggan.",
                     'rfm', ['PREP_DATA', 'ELBOW_DATA', 'KMEANS_DATA', 'KMANUAL_DATA', 'DASHBOARD_DATA'])


@app.route('/prep', methods=['POST'])
@login_required
def prep():
    if not RFM_DATA.get('success'):
        flash('Lakukan RFM terlebih dahulu.', 'error')
        return redirect(url_for('index', view='rfm'))
    return _run_step(run_preprocessing, lambda r: 'Transformasi dan standardisasi selesai.',
                     'prep', ['ELBOW_DATA', 'KMEANS_DATA', 'KMANUAL_DATA', 'DASHBOARD_DATA'])


@app.route('/elbow', methods=['POST'])
@login_required
def elbow():
    if not PREP_DATA.get('success'):
        flash('Lakukan Transformasi terlebih dahulu.', 'error')
        return redirect(url_for('index', view='prep'))
    return _run_step(run_elbow, lambda r: f"Elbow Method selesai. k optimal = {r['optimal_k']}.",
                     'elbow', ['KMEANS_DATA', 'DASHBOARD_DATA'])


@app.route('/kmeans', methods=['POST'])
@login_required
def kmeans():
    if not ELBOW_DATA.get('success'):
        flash('Lakukan Elbow Method terlebih dahulu.', 'error')
        return redirect(url_for('index', view='elbow'))
    try:
        k = int(request.form.get('k', ELBOW_DATA.get('optimal_k', 3)))
    except Exception:
        k = ELBOW_DATA.get('optimal_k', 3)
    return _run_step(run_kmeans, lambda r: f"K-Means selesai dengan k = {r['k']}.",
                     'kmeans', ['DASHBOARD_DATA'], k)


@app.route('/kmanual', methods=['POST'])
@login_required
def kmanual():
    if not PREP_DATA.get('success'):
        flash('Lakukan Transformasi terlebih dahulu.', 'error')
        return redirect(url_for('index', view='prep'))
    try:
        k = int(request.form.get('k', '').strip())
    except Exception:
        flash('Jumlah cluster (k) harus berupa bilangan bulat.', 'error')
        return redirect(url_for('index', view='kmanual'))
    k_max = max(2, min(10, len(PREP_DATA['prep_df']) - 1))
    if k < 2 or k > k_max:
        flash(f'Jumlah cluster (k) harus antara 2 dan {k_max}.', 'error')
        return redirect(url_for('index', view='kmanual'))
    return _run_step(run_kmanual, lambda r: f"K Manual selesai dengan k = {r['k']}.",
                     'kmanual', [], k)


@app.route('/dashboard', methods=['POST'])
@login_required
def dashboard():
    if not KMEANS_DATA.get('success'):
        flash('Lakukan K-Means terlebih dahulu.', 'error')
        return redirect(url_for('index', view='kmeans'))
    return _run_step(run_dashboard, lambda r: 'Dashboard BI siap.', 'dashboard', [])


def _csv_response(df, name):
    buf = io.BytesIO(df.to_csv(index=False).encode('utf-8-sig'))
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name=name)


@app.route('/download/<kind>')
@login_required
def download(kind):
    if kind == 'clean' and CLEANING_DATA.get('success'):
        return _csv_response(CLEANING_DATA['clean_df'], 'data_bersih.csv')
    if kind == 'rfm' and RFM_DATA.get('success'):
        tpl = build_rfm_template()
        tpl['Last_Transaction'] = pd.to_datetime(tpl['Last_Transaction']).dt.strftime('%Y-%m-%d')
        return _csv_response(tpl, 'tabel_rfm.csv')
    if kind == 'segment' and KMEANS_DATA.get('success'):
        return _csv_response(KMEANS_DATA['labeled_df'], 'pelanggan_tersegmentasi.csv')
    if kind == 'kmanual' and KMANUAL_DATA.get('success'):
        return _csv_response(KMANUAL_DATA['labeled_df'], 'pelanggan_k_manual.csv')
    flash('Data belum tersedia untuk diunduh.', 'error')
    return redirect(url_for('index'))


if __name__ == '__main__':
    print("=" * 60)
    print("  TOKOKITA - SEGMENTASI PELANGGAN (RFM + K-MEANS)")
    print("  Sumber data: sheet 'Data Transaksi Mentah'")
    print("  Buka: http://127.0.0.1:5000   Login: adel / adelia123")
    print("=" * 60)
    app.run(debug=True, port=5000)