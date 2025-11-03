import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

// 앱의 다른 부분에서 이 페이지를 부를 때
// Navigator.push(context, MaterialPageRoute(builder: (context) => PersonalSettingsPage()));
// 와 같이 사용할 수 있습니다.

class PersonalSettingsPage extends StatefulWidget {
  // 생성자(const) 추가
  const PersonalSettingsPage({Key? key}) : super(key: key);

  @override
  _PersonalSettingsPageState createState() => _PersonalSettingsPageState();
}

class _PersonalSettingsPageState extends State<PersonalSettingsPage> {
  // 폼 검증을 위한 글로벌 키
  final _formKey = GlobalKey<FormState>();

  // 성별 선택 (null: 선택 안됨, 0: 남성, 1: 여성, 2: 밝히기 싫음)
  int? _selectedGender;

  // 텍스트 필드 컨트롤러
  final _birthYearController = TextEditingController();
  final _ageController = TextEditingController();
  final _heightController = TextEditingController();
  final _weightController = TextEditingController();
  final _otherConditionController = TextEditingController();

  // 질병 체크박스 상태
  // (key: 고유 ID, value: 체크 여부)
  final Map<String, bool> _conditions = {
    'diabetes': false,
    'hypertension': false,
    'kidney': false,
    'allergy': false,
    'none': false,
  };

  // '없음' 체크박스 로직 처리
  void _onConditionChanged(String key, bool? value) {
    setState(() {
      if (value == null) return;

      if (key == 'none' && value) {
        // '없음'을 선택한 경우, 다른 모든 항목 해제
        _conditions.updateAll((k, v) => k == 'none' ? true : false);
        _otherConditionController.clear(); // 기타 입력란도 비움
      } else if (key != 'none' && value) {
        // 다른 항목을 선택한 경우, '없음' 해제
        _conditions[key] = value;
        _conditions['none'] = false;
      } else {
        // 체크 해제
        _conditions[key] = value;
      }
    });
  }
  
  // 기타 입력란에 입력 시 '없음' 해제
  void _onOtherConditionInput() {
    if (_otherConditionController.text.isNotEmpty) {
      setState(() {
        _conditions['none'] = false;
      });
    }
  }

  @override
  void initState() {
    super.initState();
    // 기타 입력란 리스너 추가
    _otherConditionController.addListener(_onOtherConditionInput);
  }

  @override
  void dispose() {
    // 컨트롤러 정리
    _birthYearController.dispose();
    _ageController.dispose();
    _heightController.dispose();
    _weightController.dispose();
    _otherConditionController.removeListener(_onOtherConditionInput);
    _otherConditionController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // AppBar (앱 상단 바)
      appBar: AppBar(
        title: const Text(
          '개인 설정',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        // main.dart의 테마를 따르도록 설정 (또는 여기서 직접 지정)
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      // 스크롤 가능한 본문
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          // 폼 위젯
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // 헤더 텍스트
                const Center(
                  child: Padding(
                    padding: EdgeInsets.only(bottom: 24.0),
                    child: Text(
                      '정확한 영양 분석을 위해 정보를 입력해주세요.',
                      style: TextStyle(fontSize: 16, color: Colors.black54),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),

                // 성별 선택
                _buildSectionTitle('성별'),
                _buildGenderSelector(),
                const SizedBox(height: 24),

                // 생년
                _buildSectionTitle('생년'),
                _buildTextFormField(
                  controller: _birthYearController,
                  hintText: '예: 1990',
                  suffixText: '년',
                ),
                const SizedBox(height: 24),

                // 나이
                _buildSectionTitle('나이'),
                _buildTextFormField(
                  controller: _ageController,
                  hintText: '예: 30',
                  suffixText: '세',
                ),
                const SizedBox(height: 24),

                // 키
                _buildSectionTitle('키'),
                _buildTextFormField(
                  controller: _heightController,
                  hintText: '예: 170',
                  suffixText: 'cm',
                ),
                const SizedBox(height: 24),

                // 몸무게
                _buildSectionTitle('몸무게'),
                _buildTextFormField(
                  controller: _weightController,
                  hintText: '예: 65',
                  suffixText: 'kg',
                ),
                const SizedBox(height: 32),

                // 질병 유무
                _buildSectionTitle('가지고 있는 질환 (선택)'),
                _buildConditionCheckbox('diabetes', '당뇨'),
                _buildConditionCheckbox('hypertension', '고혈압'),
                _buildConditionCheckbox('kidney', '신장 질환'),
                _buildConditionCheckbox('allergy', '식품 알레르기'),
                _buildConditionCheckbox('none', '없음'),
                
                // 기타 입력란
                Padding(
                  padding: const EdgeInsets.only(top: 8.0, left: 16.0, right: 16.0),
                  child: TextFormField(
                    controller: _otherConditionController,
                    // '없음'이 체크되면 비활성화
                    enabled: !_conditions['none']!, 
                    decoration: InputDecoration(
                      hintText: '기타 질환이나 알레르기 입력...',
                      filled: true,
                      fillColor: Colors.grey[100],
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12.0),
                        borderSide: BorderSide.none,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 32),

                // 저장 버튼
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () {
                      // 폼 검증
                      if (_formKey.currentState!.validate()) {
                        // 저장 로직 (여기서는 스낵바로 피드백)
                        // 실제 앱에서는 데이터를 수집하여 Firebase, DB 등에 저장
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('설정이 저장되었습니다!'),
                            backgroundColor: Colors.green,
                          ),
                        );
                        // 예: 
                        // String birthYear = _birthYearController.text;
                        // int gender = _selectedGender ?? -1;
                        // print('저장: 성별 $gender, 생년 $birthYear, ...');
                      }
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue.shade600,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16.0),
                      textStyle: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12.0),
                      ),
                      elevation: 5,
                    ),
                    child: const Text('저장하기'),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // 섹션 제목 위젯 빌더
  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Text(
        title,
        style: const TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
          color: Colors.black87,
        ),
      ),
    );
  }

  // 성별 선택 위젯 (ChoiceChip 사용)
  Widget _buildGenderSelector() {
    final List<String> genders = ['남성', '여성', '밝히기 싫음'];
    final List<IconData> icons = [Icons.male, Icons.female, Icons.not_interested];

    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: List<Widget>.generate(3, (index) {
        return Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4.0),
            child: ChoiceChip(
              label: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    icons[index],
                    size: 18,
                    color: _selectedGender == index
                        ? Colors.blue.shade700
                        : Colors.black54,
                  ),
                  const SizedBox(height: 4),
                  Flexible(
                    child: Text(
                      genders[index],
                      style: TextStyle(
                        fontSize: 11,
                        color: _selectedGender == index
                            ? Colors.blue.shade700
                            : Colors.black54,
                        fontWeight: _selectedGender == index
                            ? FontWeight.bold
                            : FontWeight.normal,
                      ),
                      textAlign: TextAlign.center,
                      overflow: TextOverflow.visible,
                      maxLines: 2,
                    ),
                  ),
                ],
              ),
              selected: _selectedGender == index,
              onSelected: (bool selected) {
                setState(() {
                  if (selected) {
                    _selectedGender = index;
                  }
                });
              },
              selectedColor: Colors.blue.shade50,
              backgroundColor: Colors.grey.shade100,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
                side: BorderSide(
                  color: _selectedGender == index
                      ? Colors.blue.shade500
                      : Colors.grey.shade300,
                  width: _selectedGender == index ? 2 : 1,
                ),
              ),
              padding: const EdgeInsets.symmetric(vertical: 12.0, horizontal: 4.0),
              showCheckmark: false, // 체크마크 숨기기
            ),
          ),
        );
      }),
    );
  }

  // 공통 텍스트 폼 필드 빌더
  Widget _buildTextFormField({
    required TextEditingController controller,
    required String hintText,
    required String suffixText,
  }) {
    return TextFormField(
      controller: controller,
      keyboardType: TextInputType.number, // 숫자 키패드
      // 숫자만 입력 가능하도록 설정
      inputFormatters: [FilteringTextInputFormatter.digitsOnly],
      decoration: InputDecoration(
        hintText: hintText,
        suffixText: suffixText,
        suffixStyle: const TextStyle(fontSize: 16, color: Colors.black54),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12.0),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12.0),
          borderSide: BorderSide(color: Colors.grey.shade300, width: 1.5),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12.0),
          borderSide: BorderSide(color: Colors.blue.shade500, width: 2.0),
        ),
        filled: true,
        fillColor: Colors.white,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 16.0),
      ),
      style: const TextStyle(fontSize: 16),
      // 간단한 유효성 검사 (비어있지 않게)
      validator: (value) {
        if (value == null || value.isEmpty) {
          // '생년', '나이', '키', '몸무게' 등의 정보를 title에서 가져올 수 있으나,
          // 여기서는 간단히 공통 메시지를 사용합니다.
          return '정보를 입력해주세요.'; 
        }
        return null;
      },
    );
  }

  // 질병 체크박스 위젯 빌더
  Widget _buildConditionCheckbox(String key, String title) {
    return CheckboxListTile(
      title: Text(
        title,
        style: const TextStyle(fontSize: 16),
      ),
      value: _conditions[key],
      // '없음'이 아닌 다른 항목들은 '없음'이 체크되면 비활성화
      enabled: key == 'none' ? true : !_conditions['none']!,
      onChanged: (bool? value) {
        _onConditionChanged(key, value);
      },
      controlAffinity: ListTileControlAffinity.leading, // 체크박스를 앞에 배치
      activeColor: Colors.blue.shade600,
      contentPadding: EdgeInsets.zero,
    );
  }
}
