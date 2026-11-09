// 논문 초안: 한식(AIHub 122)과 양식(Nutrition5k)에서 영양 추정 파이프라인 비교
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, Footer, PageNumber, TableOfContents, ImageRun,
} = require("docx");

const FONT = "Malgun Gothic";
const W = 9638; // A4 본문 폭 (좌우 여백 2 cm)
const OUT_DIR = "/home/leehogwang/food_b/paper";
const OUT = path.join(OUT_DIR, "한식_양식_영양추정_파이프라인_비교_초안.docx");
const FIG_DIR = path.join(__dirname, "..", "figs");

// PNG 그림 + 캡션. widthPx 는 본문 폭(약 640 px) 이하
function figure(file, caption, widthPx = 640) {
  const buf = fs.readFileSync(path.join(FIG_DIR, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height: Math.round((widthPx * h) / w) },
        altText: { title: caption.replace(/\*\*/g, ""), description: caption.replace(/\*\*/g, ""), name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 }, children: runs(caption, { size: 18 }) }),
  ];
}

// "**굵게**" 표기를 TextRun 배열로
function runs(text, base = {}) {
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) =>
    s.startsWith("**") ? new TextRun({ text: s.slice(2, -2), bold: true, font: FONT, ...base })
                       : new TextRun({ text: s, font: FONT, ...base }));
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 120, line: 320 }, alignment: o.align || AlignmentType.JUSTIFIED, ...(o.p || {}) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t), spacing: { after: 60, line: 300 } });
const N = (t) => new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(t), spacing: { after: 60, line: 300 } });
const CAP = (t) => new Paragraph({ children: runs(t, { size: 18, bold: false }), spacing: { before: 80, after: 200 }, alignment: AlignmentType.LEFT });
const TCAP = (t) => new Paragraph({ children: runs(t, { size: 19 }), spacing: { before: 200, after: 80 }, keepNext: true });

const border = { style: BorderStyle.SINGLE, size: 4, color: "A9B2BE" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths, opt = {}) {
  const sum = widths.reduce((a, b) => a + b, 0);
  const scaled = widths.map((w) => Math.round((w / sum) * W));
  scaled[scaled.length - 1] += W - scaled.reduce((a, b) => a + b, 0);
  const cell = (txt, i, head, hl) => new TableCell({
    width: { size: scaled[i], type: WidthType.DXA }, borders,
    shading: head ? { fill: "E3E8EF", type: ShadingType.CLEAR, color: "auto" } : hl ? { fill: "F3F6EA", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: String(txt).split("\n").map((line) => new Paragraph({
      children: runs(line, { size: 17, bold: head ? true : undefined }),
      alignment: i === 0 || opt.left?.includes(i) ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { after: 0, line: 260 } })),
  });
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: scaled,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, opt.hl?.includes(ri))) }))],
  });
}

const body = [];
const add = (...x) => body.push(...x);

// ───────── 표지 ─────────
add(
  new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "음식 사진 기반 영양 추정에서 데이터 도메인과 파이프라인 설계의 상호작용", font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "한식(AIHub 122)과 양식(Nutrition5k)의 파이프라인별 비교", font: FONT, size: 26, color: "4A5566" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [new TextRun({ text: "초안 · 2026-11-09 · 수치 출처는 부록 A", font: FONT, size: 19, color: "6B7685" })] }),
);

// ───────── 초록 ─────────
add(H1("초록"));
add(P("음식 사진 한 장에서 칼로리·지방·단백질·탄수화물을 추정하는 연구는 대개 한 데이터셋에서 한 구조를 개선한다. 그러나 같은 구조가 다른 음식 문화의 데이터에서도 잘 동작하는지는 거의 확인되지 않았다. 본 연구는 한식 데이터(AIHub 122, 처음 보는 음식 23종 5,884장과 아는 음식 78종 4,387장)와 양식 데이터(Nutrition5k, test 507접시)에서 성격이 다른 파이프라인 여섯 가지를 같은 지표(4대 영양소 평균 PMAE, G4)로 비교하였다."));
add(P("양식에서는 RGB-D 사진에서 영양값을 바로 회귀하는 파이프라인이 강했다. 직접 회귀와 분해형 추정을 게이트로 섞은 통합 설계가 G4 20.26으로 가장 좋았고, 학습된 게이트는 직접 회귀 쪽에 약 0.9의 비중을 두었다. 반면 한식의 처음 보는 음식에서는 같은 계열의 직접 회귀가 G4 52~57로 크게 실패하였다. 한식에서는 영양값을 \"1인분 × 양 배율\"로 나누고 1인분을 대규모 언어 모델(LLM)의 지식으로 채우는 파이프라인이 강했다. 아는 음식은 학습 라벨 표 조회로 G4 0.36, 처음 보는 음식은 음식 박스 기반 양 추정을 더해 G4 33.64 ± 0.47(시드 3개)이었다. 이 분해형 설계를 양식에 그대로 옮기면 G4 28.98로 직접 회귀보다 나빴다."));
add(P("이 차이는 두 데이터의 정답이 만들어진 방식에서 온다고 본다. 한식 정답은 \"영양 DB의 음식별 1인분 값 × 다섯 단계 양 비율\"이라 음식이 정해지면 영양 밀도가 정해지고, 사진은 양만 알려 주면 된다. 양식 정답은 접시마다 저울로 잰 실측값이고 여러 재료가 섞여 있어, \"1인분\"이라는 기준이 없고 실측 깊이로 본 부피가 직접 단서가 된다. 본 연구는 도메인별로 어떤 파이프라인이 왜 잘 되고 왜 안 되는지를 정리하고, 남은 오차가 한식에서는 처음 보는 음식의 1인분 값과 양 기준점, 양식에서는 보이지 않는 지방 재료에 몰려 있음을 보인다."));

// ───────── 1. 서론 ─────────
add(H1("1. 서론"));
add(P("식단 관리 앱에서 사용자는 사진 한 장만 찍고 영양 정보를 기대한다. 이를 위한 기존 연구는 주로 Nutrition5k 같은 양식 구내식당 데이터에서 사진(또는 RGB-D) → 영양값 회귀 모델을 개선해 왔다. 한식은 국·찌개·반찬처럼 그릇 안에 담긴 형태가 많고, 공개 데이터의 정답도 실측이 아니라 영양 DB의 1인분 값에 양 등급을 곱해 만든 경우가 많다. 따라서 양식에서 검증된 구조가 한식에서도 통한다고 가정하기 어렵다."));
add(P("본 연구의 질문은 다음 세 가지다."));
add(N("같은 파이프라인이 한식과 양식에서 어떻게 다르게 동작하는가?"));
add(N("어떤 파이프라인이 어떤 도메인에서 잘 되고, 그 이유는 무엇인가?"));
add(N("각 도메인에서 남은 오차는 어느 단계(음식 인식, 1인분 값, 양 추정)에 있는가?"));
add(P("기여는 다음과 같다. (1) 직접 회귀, 텍스트 결합, 분해형, 통합형, 라우팅형 파이프라인을 두 도메인 세 분할에서 같은 지표로 비교하였다. (2) 도메인별로 성능이 갈리는 이유를 정답 생성 방식, 깊이 정보의 출처, 텍스트(음식명·재료) 정보의 역할로 설명하였다. (3) 한식의 처음 보는 음식에 대해 오차를 단계별로 분해하고, 시도했으나 효과가 없거나 작았던 방법(외부 영양 DB 조회, LLM 양 추정, 유사 음식 참조, 음식 정체성 제거 학습)을 함께 보고하였다."));

// ───────── 2. 데이터와 지표 ─────────
add(H1("2. 데이터와 평가 지표"));
add(H2("2.1 한식: AIHub 122 양추정 데이터"));
add(P("음식별로 같은 그릇에 양을 다섯 단계(Q1~Q5)로 바꿔 담아 측면(side)과 위(top)에서 찍은 사진이다. 정답은 **영양 DB의 1인분 값 × 양 비율(Q1 0.25 · Q2 0.5 · Q3 0.75 · Q4 1.0 · Q5 1.25)**이며, 같은 음식 안에서 1인분 값은 일정하다. 1인분 값의 출처는 122 데이터에 함께 제공된 외식메뉴 400종 영양 DB(44.음식분류_AI_데이터_영양DB.xlsx)로 확인하였다. 사진마다 음식·그릇·숟가락·포크·동전 박스가 표시되어 있다."));
add(P("두 가지 분할을 쓴다. **아는 음식(122_1_forseen)**은 학습 31,484장, 검증 3,423장, test 4,387장(78종)으로 test 음식이 학습에 있다. **처음 보는 음식(122_2_forunseen)**은 학습 10,840장(44종), 검증 2,989장(12종), test 5,884장(23종)으로 test 음식이 학습에 없다."));
add(H2("2.2 양식: Nutrition5k"));
add(P("미국 구내식당(cafe1) 접시를 저울에 올려 촬영하고 재료를 하나씩 더할 때마다 무게를 잰 데이터다. 정답은 **접시 단위 실측 영양값과 재료별 그램**이다. 위에서 내려다본 RealSense RGB-D(실측 깊이)와 측면 영상이 있다. 학습 2,478접시, 검증 276접시, test 507접시를 쓴다. 음식명 대신 재료 목록을 텍스트로 쓴다."));
add(TCAP("**표 1.** 두 도메인 데이터의 성격"));
add(table(["항목", "한식 (AIHub 122)", "양식 (Nutrition5k)"], [
  ["정답이 만들어진 방식", "영양 DB 1인분 값 × 양 비율 (5단계, 이산)", "접시별 저울 실측 (연속)"],
  ["같은 음식 안의 영양 밀도", "일정 (음식이 정해지면 g당 값 고정)", "접시마다 다름 (재료 구성이 바뀜)"],
  ["한 사진의 구성", "그릇 하나에 음식 하나", "여러 재료가 섞인 접시"],
  ["깊이 정보", "없음 (UniDepth V2·Depth Anything 추정)", "RealSense 실측 깊이"],
  ["텍스트 정보", "음식명 (정답 또는 Gemini 추정)", "재료 목록 (정답 또는 Gemini 추정)"],
  ["\"1인분\" 개념", "있음 (DB 1회 제공량)", "없음"],
  ["처음 보는 음식 평가", "있음 (음식 단위 분할)", "없음 (접시 단위 분할)"],
], [2.2, 3.4, 3.4], { left: [1, 2] }));
add(H2("2.3 평가 지표"));
add(P("주 지표는 영양소별 **G = Σ|예측 − 정답| / Σ정답 × 100**(논문에서 PMAE로 부르는 정의)이고, 칼로리·지방·단백질·탄수화물 네 값을 평균해 **G4**로 쓴다. 보조로 사진별 상대 오차 평균 **A = mean(|p − t| / (t + 1)) × 100**과 그 4대 평균 A4를 쓴다. G는 오차 합 기준이라 칼로리가 큰 음식이 결과를 좌우하고, A는 작은 접시·작은 양의 상대 오차를 크게 친다."));
add(P("텍스트 조건은 세 가지다. ① 학습·평가 모두 정답 음식명(재료), ② 학습은 정답, 평가는 Gemini 추정, ③ 학습·평가 모두 Gemini 추정이다. 실사용에 해당하는 것은 ②와 ③이다."));

// ───────── 3. 파이프라인 ─────────
add(H1("3. 비교한 파이프라인"));
add(P("파이프라인은 \"사진에서 영양값을 어떻게 얻는가\"에 따라 다음 여섯 가지로 나눈다. 1~4번은 실험 40에서 같은 분할·같은 학습 예산으로 비교한 기본 구조이고, 5·6번은 이후 각 도메인에 맞춰 강화한 설계다."));
add(TCAP("**표 2.** 비교한 파이프라인"));
add(table(["기호", "이름", "구조", "영양값을 얻는 방식"], [
  ["P1", "RGB-D 직접 회귀 (ResNet)", "ResNet101-SE, RGB + 깊이 4채널", "사진 → 영양값 직접 회귀"],
  ["P2", "RGB-D 직접 회귀 (ConvNeXt)", "ConvNeXt-S, RGB + 깊이", "사진 → 영양값 직접 회귀"],
  ["P3", "텍스트 결합 멀티모달", "ConvNeXt-B 이미지 + FFT 주파수(VGG16) + Gemini 텍스트 임베딩(3072)", "음식명·재료 텍스트 + 사진 → 영양값"],
  ["P4", "분해형 (25-d)", "ConvNeXt-B + 주파수 + UniDepth 깊이 게이트 + 음식명 임베딩", "1인분 × 배율, 1인분 = LLM 사전값 + 보정 δ"],
  ["P5", "통합형 (실험 42 pipe_best)", "ConvNeXt-S/T + SE + 주파수 + 깊이 스칼라, 게이트 α", "α·직접 회귀 + (1−α)·(중량 × g당 밀도)"],
  ["P6", "라우팅형 (3번 확장, 실험 45)", "open-set 분류기 → 아는 음식 표 조회 / 모르는 음식 Gemini 1인분, 음식 박스 크롭 양 추정", "1인분(표 또는 LLM) × 양 비율"],
], [0.6, 2.0, 3.6, 3.0], { left: [1, 2, 3] }));
add(H3("P1·P2 직접 회귀"));
add(P("사진(RGB)과 깊이를 4채널로 합쳐 백본에 넣고, 회귀 헤드가 4대 영양소를 바로 낸다(그림 1). 음식명이나 1인분 같은 중간 개념이 없다. 깊이는 양식에서는 RealSense 실측값, 한식에서는 UniDepth V2나 Depth Anything 추정값을 쓴다."));
add(...figure("fig1_direct.png", "**그림 1.** P1·P2 직접 회귀 파이프라인"));
add(H3("P3과 P6의 처리 흐름"));
add(P("P3의 실사용 구성(본 문서의 \"3번\" 최고 구성)은 다음과 같다. 사진이 들어오면 ConvNeXt-Tiny 분류기와 Mahalanobis 거리 기반 open-set 판정으로 아는 음식인지 먼저 가른다. 아는 음식이면 분류기 음식명, 모르는 음식이면 Gemini 음식명을 쓴다. 이어 Gemini가 재료와 18개 대분류를 답한다. 아는 음식은 영양 모델이, 모르는 음식은 Gemini가 음식명과 재료로 1인분 값을 낸다. 양 추정 모델(ConvNeXt-Tiny + 대분류 임베딩)이 Q1~Q5를 고르고, 1인분 값에 Q 비율을 곱해 최종값을 얻는다."));
add(P("P6은 이 흐름에서 두 단계를 바꾼다. 아는 음식의 1인분은 학습 라벨에서 만든 음식별 표로 조회하고, 양 추정은 Faster R-CNN으로 찾은 음식 박스(그릇 제외)를 잘라 입력한다. 음식 박스 탐지기는 처음 보는 음식 test에서 평균 IoU 0.916이었다(그릇 박스를 그대로 쓰면 0.375). 그림 2에 두 경로와 P6에서 바뀐 부분(파란 점선과 파란 글씨)을 함께 표시하였다."));
add(...figure("fig2_routing.png", "**그림 2.** P3 라우팅 구성(3번 최고 구성)과 P6에서 바뀐 부분. 초록은 아는 음식, 주황은 모르는 음식 경로"));
add(H3("P4 분해형"));
add(P("P4는 최종값을 \"1인분 s × 배율 r\"로 나눈다(그림 3). 사진 쪽은 ConvNeXt-B, 주파수(FFT → VGG16), UniDepth 깊이 분기를 사진별 게이트로 섞어 배율 r을 낸다. 텍스트 쪽은 LLM(Gemini 3.8-flash)이 음식명만 보고 낸 1인분 사전값에서 출발해, 음식명 임베딩으로 작은 보정 δ만 학습한다. 한식 정답의 구조(DB 1인분 × 양 비율)를 그대로 모델 구조로 옮긴 셈이다."));
add(...figure("fig3_decomp.png", "**그림 3.** P4 분해형(25-d) 파이프라인"));
add(H3("P5의 게이트"));
add(P("P5는 직접 회귀 헤드와 \"예측 중량 × 100 g당 밀도\" 헤드를 함께 두고, 사진·텍스트 특징으로 게이트 α를 계산해 최종값 = α·직접 + (1−α)·분해로 섞는다(그림 4). α는 학습으로 정해지므로, 도메인이 어떤 방식을 선호하는지 보여 주는 지표가 된다."));
add(...figure("fig4_fused.png", "**그림 4.** P5 통합형(실험 42 pipe_best, 양식 설계). 한식 설계는 양 헤드(Q1~Q5)와 1인분 모델을 따로 둔다"));

// ───────── 4. 결과 ─────────
add(H1("4. 결과"));
add(H2("4.1 같은 조건에서 비교한 기본 구조 (실험 40)"));
add(P("네 기본 구조를 같은 분할·같은 학습 예산으로 학습해 비교하였다(표 3). 한식은 사진 단위 분할(아는 음식)과 음식 단위 분할(처음 보는 음식)을 썼다. 일부 한식 런은 시간 상한(약 100분)에서 멈췄다."));
add(TCAP("**표 3.** 기본 구조의 도메인별 G4 (%, 낮을수록 좋음). 칸 안은 평가 조건별 값"));
add(table(["파이프라인", "양식 N5k (507접시)", "한식 아는 음식 (사진 단위)", "한식 처음 보는 음식 (음식 단위)"], [
  ["P1 RGB-D 직접 (ResNet)", "실측 깊이 24.63\n추정 깊이 95.91\n추정 깊이 중앙값 보정 33.15", "추정 깊이 10.10", "추정 깊이 56.94"],
  ["P2 RGB-D 직접 (ConvNeXt)", "실측 깊이 24.77\n추정 깊이 27.21", "추정 깊이 7.48", "추정 깊이 52.27"],
  ["P3 텍스트 결합", "정답 텍스트 21.49\nGemini 텍스트 25.35", "정답 7.73\nGemini 9.14", "정답 49.99\nGemini 49.40"],
  ["P4 분해형 (사전값 끔)", "정답 37.50\nGemini 42.01", "정답 17.64\nGemini 25.29", "정답 51.55\nGemini 54.16"],
], [2.4, 2.6, 2.2, 2.4]));
add(P("세 가지가 두드러진다. 첫째, 양식에서는 사진만 쓰는 P2(실측 깊이 24.77)가 이미 강하고 텍스트를 더한 P3(21.49)가 가장 좋다. 둘째, 한식 처음 보는 음식에서는 모든 기본 구조가 G4 49~57로 실패한다. 셋째, LLM 사전값을 끈 분해형 P4는 양식에서 가장 나쁘다(37.50~42.01)."));
add(H2("4.2 도메인별로 강화한 파이프라인"));
add(TCAP("**표 4.** 강화한 파이프라인의 G4 (%). 별도 표기가 없으면 시드 42 한 번"));
add(table(["파이프라인", "양식 N5k", "한식 아는 음식 (forseen)", "한식 처음 보는 음식 (forunseen)"], [
  ["P3 텍스트 결합 + 라우팅 (3번 최고 구성)", "—", "1.40", "33.67\n시드 3개 34.79 ± 0.90"],
  ["P4 분해형 + LLM 사전값 (실험 41)", "① 28.98 · ② 32.42 · ③ 31.79", "① 9.13 · ② 19.07 · ③ 14.16", "① 32.93 · ② 35.58 · ③ 41.72\n시드 3개 ① 33.73 ± 1.05 · ② 36.36 ± 0.94"],
  ["P5 통합형 pipe_best (실험 42)", "① **20.26** · ② 20.48 · ③ 21.37\n(α 직접 회귀 비중 0.90~0.91)", "① 0.33 · ② 0.36 · ③ 0.50", "① 34.18 · ② 36.10 · ③ 36.29\n시드 3개 ① 34.23 ± 0.29"],
  ["P5 통합형, LLM 사전값 없음", "—", "—", "시드 3개 ① 42.72 ± 1.08 · ② 44.49 ± 0.81"],
  ["P6 라우팅 + 표 조회 + 음식 박스 Q (실험 45)", "—", "**0.36**", "**33.64 ± 0.47** (시드 3개)\nQ 확률 앙상블 32.83"],
], [3.0, 2.3, 1.9, 2.8], { hl: [2, 4] }));
add(CAP("①②③은 2.3절의 텍스트 조건이다. P3·P6의 처음 보는 음식 값은 실사용 조건(평가 때 Gemini 음식명)이다. P6 아는 음식 0.36은 표 조회 + 3번 양 추정 조합이다."));
add(P("양식 최고는 통합형 P5(20.26)이고, 한식 최고는 아는 음식에서 표 조회(0.33~0.36), 처음 보는 음식에서 라우팅형 P6(33.64 ± 0.47)이다. 같은 통합형 P5는 한식 처음 보는 음식에서 34.18로 P6과 비슷하지만, LLM 사전값을 빼면 42.72로 크게 나빠진다."));
add(H2("4.3 양식 결과의 신뢰도: 사전학습 누수"));
add(P("P2 계열과 P5의 양식 결과는 Nutrition5k 측면·위 사진으로 미리 학습한 백본(b8)에서 시작했다. 이 백본은 학습 전부터 검증 PMAE가 6.62로, 검증 접시를 이미 본 것으로 보인다. ImageNet에서 시작하면 P2는 21.16 ± 0.11에서 30.24 ± 0.68로 나빠진다. 반면 통합형 P5는 ImageNet에서 시작해도 22.49 ± 0.38(제안 설계 21.74 ± 1.11)로 크게 나빠지지 않았다. 따라서 \"양식에서 P5가 가장 좋다\"는 결론은 누수를 제거한 조건에서도 유지되지만, 20.26이라는 절대값은 낙관적이다."));

// ───────── 5. 분석 ─────────
add(H1("5. 분석: 도메인별로 무엇이 왜 잘 되고 안 되는가"));
add(H2("5.1 양식(Nutrition5k)"));
add(H3("잘 되는 파이프라인: 직접 회귀 계열(P2, P5, P3)"));
add(P("양식에서 직접 회귀가 잘 되는 이유는 정답이 사진에 보이는 물리량과 바로 이어지기 때문이라고 생각한다. 정답이 저울로 잰 접시 중량과 재료 구성이고, RealSense 실측 깊이가 담긴 양의 부피를 거의 그대로 알려 준다. 그래서 모델은 \"어떤 음식인가\"를 몰라도 \"얼마나 담겼고 무슨 색·질감인가\"로 영양값을 추정할 수 있다. 통합형 P5가 학습 끝에 직접 회귀 쪽에 α ≈ 0.90의 비중을 둔 것은, 모델 스스로 분해형보다 직접 회귀가 이 데이터에 맞다고 판단한 결과로 본다."));
add(P("재료 텍스트도 도움이 된다. P3에서 정답 재료 목록을 주면 21.49로 사진만 쓴 P2(24.77)보다 좋았다. 여러 재료가 섞인 접시에서 겉으로 보이지 않는 재료를 텍스트가 알려 주기 때문이다. 실험 43의 분해에서도 구성만 틀릴 때 G4 21.00, 양만 틀릴 때 16.31로, 양식에서는 \"무엇이 담겼나\"와 \"얼마나 담겼나\"가 둘 다 오차의 큰 몫을 차지했다."));
add(H3("안 되는 파이프라인: 분해형(P4)과 추정 깊이를 쓴 P1"));
add(P("분해형 P4는 양식에서 가장 나빴다(LLM 사전값을 켜도 28.98, 끄면 37.50). 양식에는 \"1인분\"이라는 기준이 없기 때문이라고 생각한다. P4는 1인분을 \"학습 평균 중량만큼의 영양값\"으로 억지로 정의하고 배율을 \"중량 ÷ 평균\"으로 두는데, 접시마다 재료 구성이 달라 g당 영양 밀도가 일정하지 않다. 한식에서 강점이던 \"음식이 정해지면 밀도가 정해진다\"는 가정이 양식에서는 성립하지 않는다."));
add(P("P1은 깊이 출처에 매우 민감했다. 실측 깊이 조건에서는 24.63이지만, 추정 깊이(Depth Anything) 조건에서는 95.91이었고, 중앙값으로 스케일을 맞춰도 33.15였다. 깊이의 절대 스케일이 바뀌면 크게 흔들리는 것으로 보인다. 같은 조건에서 P2(27.21)는 덜 흔들려, 사진별 표준화와 더 강한 백본이 깊이 스케일 차이를 일부 흡수한 것으로 본다."));
add(P("Gemini 재료 목록은 지방에서 특히 손해를 본다. P4에서 정답 목록 대신 Gemini 목록을 쓰면 지방 G가 33에서 42로 커졌다. 소스·드레싱·버터처럼 지방이 많은 재료는 사진에 잘 보이지 않아 LLM도 빠뜨리기 쉽다."));

add(H2("5.2 한식(AIHub 122)"));
add(H3("잘 되는 파이프라인: 분해형·라우팅형(P6, P5, P4, P3 라우팅)"));
add(P("한식에서 분해형이 잘 되는 이유는 정답 구조와 모델 구조가 같기 때문이라고 생각한다. 정답이 \"DB 1인분 × 양 비율\"이므로, 텍스트(음식명)는 1인분만, 사진은 양 등급만 맞히면 된다. 아는 음식에서는 이 구조가 극단적으로 작동한다. 학습 라벨로 음식별 1인분 표를 만들어 조회하면 1인분은 정답과 같아지고, 양 추정 정확도가 0.995라 G4가 0.33~0.36까지 내려간다. 통합형 P5에서도 분류기 경로(표 조회)로 간 사진의 G4는 0.15~0.25였다."));
add(P("처음 보는 음식에서는 외부 지식이 결정적이다. 학습 음식이 44종뿐이라, 모델만으로는 처음 보는 음식의 1인분을 알 수 없다. 3번 영양 모델은 처음 보는 음식의 1인분을 학습 평균 쪽으로 수축시켰다(음식별 정답 대비 예측 기울기: 칼로리 0.29, 지방 0.17). LLM(Gemini)이 음식명과 재료로 1인분을 주면 이 문제가 크게 줄어, 3번 구성은 45.78에서 33.67로 좋아졌다. 통합형 P5에서 LLM 사전값을 빼면 34.23에서 42.72로 나빠지는 것도 같은 이유다."));
add(H3("안 되는 파이프라인: 직접 회귀(P1, P2)와 기본 P3"));
add(P("직접 회귀는 처음 보는 음식에서 G4 52~57로 실패하였다. 한식 정답에서 영양 밀도는 음식 종류가 정하는데, 그 정보는 사진의 부피나 질감보다 \"무슨 음식인가\"에 있다. 사진만 보는 모델은 학습한 44종의 생김새와 영양값을 외우고, 처음 보는 음식에서는 비슷하게 생긴 학습 음식의 값을 따라간다. 게다가 한식에는 실측 깊이가 없어 추정 깊이를 써야 하고, 국물 요리처럼 그릇 높이로 양이 정해지는 음식이 많아 깊이 단서도 약하다."));
add(P("텍스트를 더한 기본 P3도 처음 보는 음식에서 49.40~49.99로 직접 회귀와 큰 차이가 없었다. 정답 음식명을 줘도 거의 나아지지 않아, 문제는 음식명을 아느냐가 아니라 \"처음 보는 음식명에서 1인분 값으로 일반화하지 못하는 것\"이다. 실험 41에서 학습 음식명을 Gemini 추정으로 바꾸면 아는 음식은 19.07에서 14.16으로 좋아졌지만 처음 보는 음식은 35.58에서 41.72로 나빠졌다. 학습 음식이 적은 상태에서 이름이 흔들리면 음식별 1인분을 배울 근거가 더 줄기 때문으로 본다."));

add(H2("5.3 요약: 도메인 × 파이프라인"));
add(TCAP("**표 5.** 도메인별로 잘 되는 파이프라인과 그 이유"));
add(table(["파이프라인", "양식 N5k", "한식 아는 음식", "한식 처음 보는 음식"], [
  ["P1·P2 직접 회귀", "**좋음** (24.6~24.8)\n실측 깊이가 부피를 알려 줌\n단 P1은 추정 깊이에 취약", "보통 (7.5~10.1)\n같은 음식이 학습에 있어 외워서 맞힘", "**나쁨** (52~57)\n영양 밀도가 음식 종류에 달렸는데 사진으로는 모름"],
  ["P3 텍스트 결합", "**좋음** (21.49)\n재료 텍스트가 보이지 않는 재료를 알려 줌", "좋음 (7.7, 라우팅 시 1.40)", "기본은 나쁨 (49.4)\nLLM 1인분 + 라우팅 시 33.67"],
  ["P4 분해형", "**나쁨** (28.98~42.01)\n\"1인분\" 기준이 없고 밀도가 접시마다 다름", "좋음 (9.13, 정답 이름)\nGemini 이름 학습 시 14.16", "좋음 (32.93~35.58)\n정답 구조와 같음, LLM 사전값이 핵심"],
  ["P5 통합형", "**가장 좋음** (20.26)\n게이트가 직접 회귀 선택 (α≈0.9)", "**가장 좋음** (0.33)", "좋음 (34.18)\n사전값 없으면 42.72"],
  ["P6 라우팅 + 표 조회", "해당 없음", "**가장 좋음** (0.36)", "**가장 좋음** (33.64 ± 0.47)"],
], [1.8, 2.8, 2.4, 3.0], { left: [1, 2, 3] }));

add(H2("5.4 한식 처음 보는 음식의 남은 오차"));
add(P("3번 구성에서 단계별로 정답을 넣어 오차를 분해하면, 음식명을 정답으로 바꿔도 44.86, 양을 정답으로 바꿔도 45.02였지만 1인분 값을 정답으로 바꾸면 15.76이었다(LLM 1인분 도입 전, 기준 45.78). 현재 구성(33.67)에서 양만 정답으로 바꾸면 27.93이다. 즉 남은 오차는 1인분 값이 가장 크고, 양 추정이 그다음이다."));
add(H3("1인분 값"));
add(B("공개 영양 DB는 122 정답과 기준이 다르다. 052 통합 영양성분 DB와 음식명을 맞춘 63종 중 27종은 1인분 중량 정의부터 달랐다(예: 감자탕 정답 900 g / DB 530 g). 모르는 음식의 1인분을 이 DB로 바꾸면 G4가 33.67에서 42.18로 나빠졌다."));
add(B("분류기가 가장 비슷하다고 본 아는 음식의 1인분을 빌려 오는 방법은 60.88로 크게 나빴다. 분류기는 영양이 아니라 겉모습이 비슷한 음식을 고른다(미소된장국 → 닭죽, 38 kcal → 1,064 kcal). 아는 음식 44종 중 정답과 가장 가까운 음식을 골라도 상한이 28.31이었다."));
add(B("122 정답 자체에도 일관성이 낮은 음식이 있다. 김치볶음밥은 657 kcal인데 3대 영양소로 계산하면 약 400 kcal다. 이런 음식은 어떤 외부 지식으로도 맞출 수 없다."));
add(H3("양 추정"));
add(B("양 추정 모델은 같은 음식 안의 양 순서를 거의 맞힌다(순위상관 0.83~0.97). 틀리는 대부분은 음식마다 한쪽으로 치우친 등급이다(갈치조림 −1.58등급). 모델이 처음 보는 음식의 \"1인분이 어느 정도 양인지\"라는 기준점을 모르기 때문이다."));
add(B("Gemini에게 사진으로 양 등급을 물으면 정확도가 0.543에서 0.271로 떨어졌다. Gemini는 23종 모두에서 양을 적게 보았고 Q5를 한 번도 답하지 않았다."));
add(B("특징에서 음식 정보를 지우는 적대적 학습(GRL)과 강한 색·질감 증강을 함께 쓰면 양 정확도가 0.557에서 0.589로, 가장 큰 음식별 치우침이 1.58에서 0.97등급으로 줄었다. 그러나 G4는 0.26%p만 줄었고, λ를 바꾸거나 GRL만 쓴 설정에서는 재현되지 않았다. 확률 기대값으로 연속 비율을 쓰면 G4가 0.24%p 줄었지만, 음식 단위 bootstrap 95% 구간이 0을 포함했다."));
add(B("음식 박스를 잘라 양 추정에 넣은 P6은 양 정확도를 0.522에서 0.568로 올려, G4를 34.79 ± 0.90에서 33.64 ± 0.47로 낮추고 시드 간 편차도 줄였다."));

add(H2("5.5 두 도메인에 공통인 관찰"));
add(B("**지방이 가장 어렵다.** 한식 처음 보는 음식에서 지방 G는 46~53, 양식에서도 22~42로 모든 조건에서 가장 나빴다. 기름·소스는 사진에 잘 보이지 않고, LLM의 지방 추정도 가장 흔들린다."));
add(B("**텍스트 추정 오류의 비용은 도메인마다 다르다.** 양식에서 Gemini 재료로 바꾼 손해는 P5 기준 0.2~1.1%p로 작았다. 한식 아는 음식에서는 Gemini 음식명이 정답과 정확히 같은 사진이 51%뿐이라 손해가 컸다(9.13 → 19.07). 이는 표 조회나 분류기 라우팅으로 줄일 수 있었다."));
add(B("**도메인이 선호하는 구조는 정답 생성 방식이 정한다.** 실측 정답(양식)은 직접 회귀를, DB 1인분 × 비율 정답(한식)은 분해형을 선호하였다. 통합형 P5의 게이트가 이를 스스로 따라갔다는 점에서, 두 방식을 섞어 데이터가 고르게 하는 설계가 도메인을 넘는 기본값으로 적절해 보인다."));

// ───────── 6. 한계 ─────────
add(H1("6. 한계"));
add(B("양식 결과 중 b8 백본에서 시작한 값은 검증 접시 누수 가능성이 있다(4.3절). ImageNet 시작 결과도 함께 보고해야 한다."));
add(B("한식 아는 음식의 G4 0.3~1.4는 정답이 음식별로 일정한 데이터 구조 때문이다. 실제 식당 사진에서는 같은 음식도 레시피와 양이 달라 이 수치가 그대로 나오지 않는다. 또한 같은 번호의 측면·위 사진이 학습과 test에 나뉘어 들어간 쌍이 3,196개 있다."));
add(B("한식 처음 보는 음식 test는 23종 한 번이다. 1%p 미만의 차이는 음식 구성에 따라 바뀔 수 있어, 음식 단위 K-fold 평가가 필요하다."));
add(B("두 도메인의 분할 방식이 다르다. 양식에는 처음 보는 음식 분할이 없어, \"처음 보는 음식\" 일반화를 양식에서 같은 방식으로 비교하지 못하였다."));
add(B("LLM 결과는 Gemini 모델 버전(3.5-flash-lite, 3.8-flash)과 프롬프트에 의존한다. 일부 실험은 시드 한 번이다."));

// ───────── 7. 결론 ─────────
add(H1("7. 결론"));
add(P("같은 영양 추정 파이프라인이 한식과 양식에서 정반대의 결과를 냈다. 양식에서는 실측 깊이와 재료 텍스트를 쓰는 직접 회귀 계열이 강했고(통합형 20.26, 누수 없는 시작 22.49 ± 0.38), \"1인분 × 배율\" 분해형은 1인분 기준이 없어 약했다(28.98). 한식에서는 반대로 분해형과 라우팅형이 강했고(처음 보는 음식 33.64 ± 0.47, 아는 음식 0.36), 직접 회귀는 처음 보는 음식에서 실패하였다(52~57). 이 차이는 각 데이터의 정답이 만들어진 방식, 즉 실측인가 DB 1인분 × 비율인가에서 온다고 본다."));
add(P("한식의 다음 과제는 처음 보는 음식의 1인분 값과 양 기준점이다. 1인분 값을 정답으로 주면 G4가 15.76까지, 양을 정답으로 주면 27.93까지 내려가므로 여지가 크다. 양식의 다음 과제는 보이지 않는 지방 재료와, 누수 없는 조건에서의 재검증이다. 서로 다른 음식 문화의 데이터로 영양 추정 모델을 평가할 때에는 성능 수치와 함께 정답 생성 방식과 분할 방식을 반드시 함께 보고해야 한다."));

// ───────── 부록 ─────────
add(H1("부록 A. 수치 출처"));
add(table(["내용", "경로"], [
  ["기본 구조 비교 (표 3)", "/home/leehogwang/Food-Nutrition-Estimation/experiments/40_unified/results.md"],
  ["P4 분해형 + LLM 사전값 (실험 41)", "/home/leehogwang/Food-Nutrition-Estimation/experiments/41_25d_splits/RESULT.md"],
  ["P5 통합형 양식 학습 코드", "/home/leehogwang/Food-Nutrition-Estimation/experiments/42_fused/src/train_n5k.py"],
  ["P5 양식 가중치 · 결과 (20.26)", "/home/leehogwang/Food-Nutrition-Estimation/experiments/42_fused/runs/n5k_best_gt/best.pth\n/home/leehogwang/Food-Nutrition-Estimation/experiments/42_fused/runs/n5k_best_gt/result.json"],
  ["P5 한식 결과", "/home/leehogwang/Food-Nutrition-Estimation/experiments/42_fused/results.md"],
  ["시드 반복 · 누수 점검", "/home/leehogwang/Food-Nutrition-Estimation/experiments/43_seeds/results43.md"],
  ["P6 음식 박스 탐지 · 박스 예측 · Q 학습", "/home/leehogwang/Food-Nutrition-Estimation/experiments/45_food_bbox/src/train_det.py\n…/45_food_bbox/src/predict_all.py\n…/45_food_bbox/src/quantity_foodbox.py"],
  ["P6 Q 모델 구조 · 1인분·라우팅", "/home/leehogwang/Food-Nutrition-Estimation/experiments/42_fused/src/quantity.py\n…/42_fused/src/combine_r3.py, …/43_seeds/r3exp"],
  ["P6 최종 조합 · 결과 (33.64, 0.36)", "/home/leehogwang/Food-Nutrition-Estimation/experiments/45_food_bbox/src/kr_combine_foodbox.py\n…/45_food_bbox/kr_combine_foodbox.json"],
  ["P3 3번 최고 구성 (33.67)", "/home/leehogwang/food_b/experiments/split_txt_3runs/run_pipe_osr.sh\n…/src/gemini_serving.py, …/src/eval_gemini_serving.py\n…/gemini_serving_eval.json"],
  ["5.4절 추가 분석 (052 DB, Gemini 양, 유사 음식, GRL, 기대값)", "/home/leehogwang/food_b/experiments/split_txt_3runs/QUANTITY_IMPROVEMENT_NOTES.md\n…/gemini_quantity_eval_sample5.json, …/similar_food_eval.json"],
], [3.0, 7.0], { left: [1] }));
add(P(""));
add(P("※ 이 문서는 초안이다. 표의 수치는 위 결과 파일에서 옮겼고, \"이유\"로 적은 내용은 저자의 해석이므로 검증 실험(음식 단위 K-fold, 누수 없는 양식 재학습)으로 보완해야 한다.", { run: { size: 18, color: "6B7685" } }));

const doc = new Document({
  creator: "leehogwang", title: "한식·양식 영양 추정 파이프라인 비교 (초안)",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 36, bold: true, color: "1B2430" }, paragraph: { spacing: { before: 600, after: 160 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: "1B2430" }, paragraph: { spacing: { before: 360, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 25, bold: true, color: "2B3A52" }, paragraph: { spacing: { before: 260, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 22, bold: true, color: "3B4A60" }, paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 280 } } } },
                                  { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 280 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 320 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "6B7685" })] })] }) },
    children: body,
  }],
});

fs.mkdirSync(OUT_DIR, { recursive: true });
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log(OUT); });
