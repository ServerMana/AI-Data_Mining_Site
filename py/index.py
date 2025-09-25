import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Streamlit Cloud를 위한 백엔드 설정
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import plotly.express as px
import plotly.graph_objects as go
import platform

# 한글 폰트 설정 (클라우드 환경 대응)
def setup_korean_font():
    """한국어 폰트 설정"""
    try:
        if platform.system() == 'Windows':
            # Windows 환경
            font_path = 'c:/Windows/Fonts/malgun.ttf'
            if os.path.exists(font_path):
                font_name = font_manager.FontProperties(fname=font_path).get_name()
                plt.rc('font', family=font_name)
                return True
        
        # Linux/클라우드 환경에서 시스템 폰트 찾기
        available_fonts = [f.name for f in font_manager.fontManager.ttflist]
        
        # 한글 지원 폰트 우선순위
        korean_fonts = ['NanumGothic', 'Noto Sans CJK KR', 'Malgun Gothic', 'AppleGothic', 'DejaVu Sans']
        
        for font in korean_fonts:
            if any(font.lower() in f.lower() for f in available_fonts):
                plt.rc('font', family=font)
                return True
        
        # 마지막 수단: 기본 폰트
        plt.rc('font', family='DejaVu Sans')
        return False
        
    except Exception as e:
        # 오류 발생 시 기본 폰트 사용
        plt.rc('font', family='DejaVu Sans')
        return False

# 폰트 설정 실행
import os
setup_korean_font()
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
        # 로컬과 클라우드 환경 모두 지원
        try:
            df = pd.read_excel("data/경제활동.xlsx")  # 클라우드 환경용
        except FileNotFoundError:
            df = pd.read_excel("../data/경제활동.xlsx")  # 로컬 환경용
        return df
    except FileNotFoundError:
        st.error("경제활동.xlsx 파일을 찾을 수 없습니다. data/ 폴더에 파일이 있는지 확인해주세요.")
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

def get_table_colors(style):
    """테이블 스타일에 따른 색상 반환"""
    if style == "컬러풀":
        return {
            'header_color': 'lightblue',
            'cell_color': 'white',
            'border_color': 'lightgray'
        }
    elif style == "모던":
        return {
            'header_color': 'darkslategray',
            'cell_color': 'whitesmoke',
            'border_color': 'gray'
        }
    elif style == "미니멀":
        return {
            'header_color': 'white',
            'cell_color': 'white',
            'border_color': 'lightgray'
        }
    else:  # 기본
        return {
            'header_color': 'lightblue',
            'cell_color': 'white',
            'border_color': 'lightgray'
        }

# 데이터 로드
df = load_data()

if df is not None:
    # 사이드바에서 분석 옵션 선택
    st.sidebar.header("🎛️ 분석 옵션")
    
    analysis_type = st.sidebar.selectbox(
        "분석 유형 선택",
        ["전국 실업률 추이", "지역별 비교 분석", "연도별 상세 분석", "원본 데이터 보기"]
    )
    
    # 전역 설정 옵션
    st.sidebar.header("⚙️ 표시 설정")
    
    # 차트 테마 선택
    chart_theme = st.sidebar.selectbox(
        "차트 테마",
        ["plotly", "ggplot2", "seaborn", "simple_white", "plotly_dark"],
        index=0
    )
    
    # 컬러 팔레트 선택
    color_palette = st.sidebar.selectbox(
        "색상 팔레트",
        ["Set1", "Set2", "Pastel1", "Dark2", "Viridis", "Plasma"],
        index=0
    )
    
    # 표 스타일 선택
    table_style = st.sidebar.selectbox(
        "표 스타일",
        ["기본", "컬러풀", "모던", "미니멀"],
        index=0
    )
    
    # 데이터 요약 정보
    st.sidebar.header("📊 데이터 정보")
    st.sidebar.info(f"""
    **전체 데이터 개수:** {len(df):,}개 행
    **분석 기간:** {df['시점'].min()} - {df['시점'].max()}
    **분석 항목:** {len(df['항목'].unique())}개
    """)
    
    # 실시간 필터링 옵션
    st.sidebar.header("🔍 실시간 필터")
    
    # 연도 범위 슬라이더
    min_year = int(df['시점'].min())
    max_year = int(df['시점'].max())
    year_range = st.sidebar.slider(
        "연도 범위 선택",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    # 선택한 연도 범위로 데이터 필터링
    df_filtered = df[(df['시점'] >= year_range[0]) & (df['시점'] <= year_range[1])]
    
    # 필터링된 데이터 정보 표시
    if len(df_filtered) != len(df):
        st.sidebar.success(f"필터 적용됨: {len(df_filtered):,}개 행")
    
    # 필터링된 데이터를 메인 데이터로 사용
    df = df_filtered
    
    # 차트 설정을 전역 변수로 저장
    if 'chart_settings' not in st.session_state:
        st.session_state.chart_settings = {
            'theme': chart_theme,
            'palette': color_palette,
            'table_style': table_style
        }
    else:
        st.session_state.chart_settings.update({
            'theme': chart_theme,
            'palette': color_palette,
            'table_style': table_style
        })
    
    if analysis_type == "전국 실업률 추이":
        st.header("🔍 전국 실업률 추이 분석")
        
        # 실업률 데이터 처리
        unemployment_data = process_unemployment_data(df)
        
        if not unemployment_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 실업률 추이 그래프")
                
                # Plotly를 사용한 차트 (테마 적용)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=unemployment_data['연도'],
                    y=unemployment_data['실업률(%)'],
                    mode='lines+markers',
                    line=dict(color='red', width=3),
                    marker=dict(size=8, color='red'),
                    text=[f'{y:.1f}%' for y in unemployment_data['실업률(%)']],
                    textposition='top center',
                    textfont=dict(size=12),
                    name='실업률'
                ))
                
                fig.update_layout(
                    title='연도별 전국 실업률 추이',
                    xaxis_title='연도',
                    yaxis_title='실업률(%)',
                    font=dict(size=12),
                    showlegend=False,
                    height=500,
                    hovermode='x',
                    template=st.session_state.chart_settings['theme']
                )
                
                fig.update_traces(texttemplate='%{text}', textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 실업률 데이터 테이블")
                display_data = unemployment_data[['연도', '실업률(%)']].copy()
                display_data['실업률(%)'] = display_data['실업률(%)'].round(2)
                
                # 선택한 스타일 적용
                table_colors = get_table_colors(st.session_state.chart_settings['table_style'])
                
                # 인터랙티브한 Plotly 테이블
                fig_table = go.Figure(data=[go.Table(
                    header=dict(
                        values=['<b>연도</b>', '<b>실업률(%)</b>'],
                        fill_color=table_colors['header_color'],
                        align='center',
                        font=dict(size=14, color='black'),
                        line=dict(color=table_colors['border_color'], width=1)
                    ),
                    cells=dict(
                        values=[display_data['연도'].tolist(), 
                               [f"{rate:.2f}" for rate in display_data['실업률(%)'].tolist()]],
                        fill_color=table_colors['cell_color'],
                        align='center',
                        font=dict(size=12),
                        height=30,
                        line=dict(color=table_colors['border_color'], width=1)
                    )
                )])
                
                fig_table.update_layout(
                    title="연도별 실업률 상세 데이터",
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    template=st.session_state.chart_settings['theme']
                )
                
                st.plotly_chart(fig_table, use_container_width=True)
                
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
                    
                    # Plotly 차트로 변경 (색상 팔레트 적용)
                    fig = px.line(
                        plot_df_filtered, 
                        x='연도', 
                        y='값', 
                        color='지역',
                        title=f'{metric} 지역별 추이 (상위 5개 지역)',
                        markers=True,
                        color_discrete_sequence=px.colors.qualitative.__dict__[st.session_state.chart_settings['palette']]
                    )
                    
                    fig.update_layout(
                        xaxis_title='연도',
                        yaxis_title=metric,
                        font=dict(size=12),
                        height=500,
                        hovermode='x unified',
                        template=st.session_state.chart_settings['theme']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
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
                
                # 인터랙티브한 지역별 데이터 테이블
                if not table_data.empty:
                    # 테이블 헤더 준비
                    headers = ['<b>연도</b>'] + [f'<b>{region}</b>' for region in regions]
                    
                    # 테이블 데이터 준비
                    table_values = [table_data['시점'].tolist()]
                    for region in regions:
                        values = []
                        for val in table_data[region]:
                            if pd.isna(val):
                                values.append('-')
                            else:
                                values.append(f"{val:,.0f}")
                        table_values.append(values)
                    
                    # Plotly 테이블 생성
                    fig_table = go.Figure(data=[go.Table(
                        header=dict(
                            values=headers,
                            fill_color='lightgreen',
                            align='center',
                            font=dict(size=12, color='black'),
                            height=40
                        ),
                        cells=dict(
                            values=table_values,
                            fill_color='white',
                            align='center',
                            font=dict(size=10),
                            height=30
                        )
                    )])
                    
                    fig_table.update_layout(
                        title=f"{metric} 지역별 상세 데이터",
                        height=500,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    st.plotly_chart(fig_table, use_container_width=True)
    
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
                        
                        # 인터랙티브한 상위 지역 테이블
                        if top_regions:
                            region_df = pd.DataFrame(top_regions, columns=['지역', '값'])
                            region_df['순위'] = range(1, len(region_df) + 1)
                            region_df = region_df[['순위', '지역', '값']]
                            
                            fig_rank = go.Figure(data=[go.Table(
                                header=dict(
                                    values=['<b>순위</b>', '<b>지역</b>', '<b>값</b>'],
                                    fill_color='lightcoral',
                                    align='center',
                                    font=dict(size=12, color='black')
                                ),
                                cells=dict(
                                    values=[
                                        region_df['순위'].tolist(),
                                        region_df['지역'].tolist(),
                                        [f"{val:,.0f}" for val in region_df['값'].tolist()]
                                    ],
                                    fill_color='white',
                                    align='center',
                                    font=dict(size=11),
                                    height=35
                                )
                            )])
                            
                            fig_rank.update_layout(
                                title=f"{metric} 상위 5개 지역",
                                height=300,
                                margin=dict(l=0, r=0, t=30, b=0)
                            )
                            
                            st.plotly_chart(fig_rank, use_container_width=True)
    
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
        
        # 인터랙티브한 필터링 옵션 추가
        col3, col4 = st.columns(2)
        
        with col3:
            # 페이지네이션을 위한 행 수 선택
            rows_per_page = st.selectbox("페이지당 행 수", [10, 25, 50, 100], index=1)
        
        with col4:
            # 정렬 옵션
            sort_column = st.selectbox("정렬 기준 열", filtered_df.columns.tolist())
            sort_ascending = st.checkbox("오름차순 정렬", value=True)
        
        # 데이터 정렬
        if sort_column:
            filtered_df = filtered_df.sort_values(by=sort_column, ascending=sort_ascending)
        
        # 페이지네이션
        total_rows = len(filtered_df)
        total_pages = (total_rows - 1) // rows_per_page + 1
        
        if total_pages > 1:
            page_number = st.selectbox(f"페이지 선택 (총 {total_pages}페이지)", 
                                     range(1, total_pages + 1))
            start_idx = (page_number - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)
            page_data = filtered_df.iloc[start_idx:end_idx]
            
            st.write(f"📄 페이지 {page_number}/{total_pages} (행 {start_idx+1}-{end_idx}/{total_rows})")
        else:
            page_data = filtered_df
        
        # 고급 데이터프레임 표시 (편집 가능)
        edited_df = st.data_editor(
            page_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                col: st.column_config.NumberColumn(
                    format="%.2f" if col in ['계'] else None
                ) for col in page_data.columns if page_data[col].dtype in ['int64', 'float64']
            }
        )
        
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
