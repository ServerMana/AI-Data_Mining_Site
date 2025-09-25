import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname='c:/Windows/Fonts/malgun.ttf').get_name()
    plt.rc('font', family=font_name)
else:
    plt.rc('font', family='AppleGothic')  # Mac 예시
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 페이지 설정
st.set_page_config(
    page_title="한국 경제활동 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

st.title("📊 한국 경제활동 분석 대시보드")
st.markdown("---")

@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    try:
        df = pd.read_excel("../data/경제활동.xlsx")
        return df
    except FileNotFoundError:
        st.error("../data/경제활동.xlsx 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return None

def process_unemployment_data(df):
    """실업률 데이터 처리"""
    # 실업자 및 경제활동인구 데이터 선택
    unemp = df[df['항목'] == '실업자 (천명)'].copy()
    pop = df[df['항목'] == '경제활동인구 (천명)'].copy()
    
    # 결측치 제거
    unemp = unemp[unemp['계'].notnull()]
    pop = pop[pop['계'].notnull()]
    
    # 인덱스 초기화
    unemp = unemp.reset_index(drop=True)
    pop = pop.reset_index(drop=True)
    
    # 연도(시점) 기준으로 merge
    merged = pd.merge(
        unemp[['시점', '계']],
        pop[['시점', '계']],
        on='시점',
        suffixes=('_실업자', '_경제활동')
    )
    
    # '계' 컬럼 숫자형 변환 (쉼표 제거)
    merged['계_실업자'] = pd.to_numeric(merged['계_실업자'].astype(str).str.replace(',', ''), errors='coerce')
    merged['계_경제활동'] = pd.to_numeric(merged['계_경제활동'].astype(str).str.replace(',', ''), errors='coerce')
    
    # 실업률 계산
    merged['실업률(%)'] = (merged['계_실업자'] / merged['계_경제활동']) * 100
    
    # 결과 데이터프레임
    result = merged[['시점', '실업률(%)', '계_실업자', '계_경제활동']].rename(columns={'시점': '연도'})
    
    # 결측치 제거 및 정렬
    result = result.dropna(subset=['연도', '실업률(%)'])
    result['연도'] = pd.to_numeric(result['연도'], errors='coerce')
    result = result.sort_values('연도')
    
    return result

def get_regional_data(df, metric, years):
    """지역별 데이터 가져오기"""
    regional_data = df[df['항목'] == metric].copy()
    
    # 지역 컬럼 (계 제외)
    regions = [col for col in regional_data.columns if col not in ['항목', '시점', '계']]
    
    # 연도 필터링
    if years:
        regional_data = regional_data[regional_data['시점'].isin(years)]
    
    return regional_data, regions

# 데이터 로드
df = load_data()

if df is not None:
    # 사이드바에서 분석 옵션 선택
    st.sidebar.header("분석 옵션")
    
    analysis_type = st.sidebar.selectbox(
        "분석 유형 선택",
        ["전국 실업률 추이", "지역별 비교 분석", "연도별 상세 분석", "원본 데이터 보기"]
    )
    
    if analysis_type == "전국 실업률 추이":
        st.header("🔍 전국 실업률 추이 분석")
        
        # 실업률 데이터 처리
        unemployment_data = process_unemployment_data(df)
        
        if not unemployment_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 실업률 추이 그래프")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(unemployment_data['연도'], unemployment_data['실업률(%)'], 
                       marker='o', color='red', linewidth=2, markersize=8)
                ax.set_title('연도별 전국 실업률 추이', fontsize=16, fontweight='bold')
                ax.set_xlabel('연도', fontsize=12)
                ax.set_ylabel('실업률(%)', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # 실업률 값 표시
                for x, y in zip(unemployment_data['연도'], unemployment_data['실업률(%)']):
                    ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                              xytext=(0,10), ha='center', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("📊 실업률 데이터 테이블")
                display_data = unemployment_data[['연도', '실업률(%)']].copy()
                display_data['실업률(%)'] = display_data['실업률(%)'].round(2)
                st.dataframe(display_data, use_container_width=True)
                
                # 통계 정보
                st.subheader("📈 주요 통계")
                avg_rate = unemployment_data['실업률(%)'].mean()
                max_rate = unemployment_data['실업률(%)'].max()
                min_rate = unemployment_data['실업률(%)'].min()
                
                st.metric("평균 실업률", f"{avg_rate:.2f}%")
                st.metric("최고 실업률", f"{max_rate:.2f}%")
                st.metric("최저 실업률", f"{min_rate:.2f}%")
    
    elif analysis_type == "지역별 비교 분석":
        st.header("🗺️ 지역별 비교 분석")
        
        # 분석 지표 선택
        metric = st.selectbox(
            "분석 지표 선택",
            ["경제활동인구 (천명)", "취업자 (천명)", "실업자 (천명)"]
        )
        
        # 연도 선택
        available_years = sorted(df['시점'].unique())
        selected_years = st.multiselect(
            "분석할 연도 선택 (전체 선택하려면 비워두세요)",
            available_years,
            default=available_years[-2:]  # 최근 2년 기본 선택
        )
        
        if not selected_years:
            selected_years = available_years
        
        regional_data, regions = get_regional_data(df, metric, selected_years)
        
        if not regional_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📊 {metric} 지역별 비교")
                
                # 지역별 데이터 준비
                plot_data = []
                for _, row in regional_data.iterrows():
                    for region in regions:
                        if pd.notnull(row[region]):
                            plot_data.append({
                                '연도': row['시점'],
                                '지역': region,
                                '값': pd.to_numeric(str(row[region]).replace(',', ''), errors='coerce')
                            })
                
                plot_df = pd.DataFrame(plot_data)
                plot_df = plot_df.dropna()
                
                if not plot_df.empty:
                    # 상위 5개 지역만 표시
                    top_regions = plot_df.groupby('지역')['값'].mean().nlargest(5).index
                    plot_df_filtered = plot_df[plot_df['지역'].isin(top_regions)]
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    for region in top_regions:
                        region_data = plot_df_filtered[plot_df_filtered['지역'] == region]
                        ax.plot(region_data['연도'], region_data['값'], 
                               marker='o', label=region, linewidth=2)
                    
                    ax.set_title(f'{metric} 지역별 추이 (상위 5개 지역)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('연도', fontsize=12)
                    ax.set_ylabel(f'{metric}', fontsize=12)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with col2:
                st.subheader("📋 지역별 데이터 테이블")
                
                # 데이터 테이블 준비
                table_data = regional_data[['시점'] + regions].copy()
                
                # 숫자형 변환
                for region in regions:
                    table_data[region] = pd.to_numeric(
                        table_data[region].astype(str).str.replace(',', ''), 
                        errors='coerce'
                    )
                
                st.dataframe(table_data, use_container_width=True)
    
    elif analysis_type == "연도별 상세 분석":
        st.header("📅 연도별 상세 분석")
        
        # 연도 선택
        available_years = sorted(df['시점'].unique())
        selected_year = st.selectbox("분석할 연도 선택", available_years, index=len(available_years)-1)
        
        year_data = df[df['시점'] == selected_year]
        
        if not year_data.empty:
            col1, col2, col3 = st.columns(3)
            
            # 각 지표별 데이터 추출
            metrics = ['경제활동인구 (천명)', '취업자 (천명)', '실업자 (천명)']
            
            for i, metric in enumerate(metrics):
                metric_data = year_data[year_data['항목'] == metric]
                
                if not metric_data.empty:
                    with [col1, col2, col3][i]:
                        st.subheader(f"📊 {metric}")
                        
                        # 전국 계 값
                        total_value = metric_data.iloc[0]['계']
                        if pd.notnull(total_value):
                            st.metric("전국", f"{total_value:,}")
                        
                        # 지역별 상위 5개
                        regions = [col for col in metric_data.columns if col not in ['항목', '시점', '계']]
                        region_values = []
                        
                        for region in regions:
                            value = metric_data.iloc[0][region]
                            if pd.notnull(value):
                                numeric_value = pd.to_numeric(str(value).replace(',', ''), errors='coerce')
                                if pd.notnull(numeric_value):
                                    region_values.append((region, numeric_value))
                        
                        # 상위 5개 지역
                        top_regions = sorted(region_values, key=lambda x: x[1], reverse=True)[:5]
                        
                        st.write("**상위 5개 지역:**")
                        for j, (region, value) in enumerate(top_regions):
                            st.write(f"{j+1}. {region}: {value:,.0f}")
    
    elif analysis_type == "원본 데이터 보기":
        st.header("📋 원본 데이터")
        
        # 필터 옵션
        col1, col2 = st.columns(2)
        
        with col1:
            available_items = df['항목'].unique()
            selected_items = st.multiselect("항목 필터", available_items, default=available_items)
        
        with col2:
            available_years = sorted(df['시점'].unique())
            selected_years = st.multiselect("연도 필터", available_years, default=available_years)
        
        # 필터링된 데이터
        filtered_df = df[
            (df['항목'].isin(selected_items)) & 
            (df['시점'].isin(selected_years))
        ]
        
        st.subheader(f"📊 필터링된 데이터 ({len(filtered_df)}개 행)")
        st.dataframe(filtered_df, use_container_width=True)
        
        # 데이터 다운로드
        csv = filtered_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="CSV 파일로 다운로드",
            data=csv,
            file_name='경제활동_데이터.csv',
            mime='text/csv'
        )

else:
    st.error("데이터를 불러올 수 없습니다. data/경제활동.xlsx 파일이 존재하는지 확인해주세요.")
    
    # 파일 업로드 옵션
    st.subheader("📁 파일 업로드")
    uploaded_file = st.file_uploader("Excel 파일을 업로드하세요", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            st.success("파일이 성공적으로 업로드되었습니다!")
            st.dataframe(df_uploaded.head())
        except Exception as e:
            st.error(f"파일 읽기 중 오류가 발생했습니다: {e}")

# 푸터
st.markdown("---")
st.markdown("📊 **한국 경제활동 분석 대시보드** | Made with Streamlit")
