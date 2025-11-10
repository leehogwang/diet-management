import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'image_processor.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '이미지 색감 보정 앱',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const ImageAugmentationPage(),
    );
  }
}

class ImageAugmentationPage extends StatefulWidget {
  const ImageAugmentationPage({super.key});

  @override
  State<ImageAugmentationPage> createState() => _ImageAugmentationPageState();
}

class _ImageAugmentationPageState extends State<ImageAugmentationPage> {
  final ImagePicker _picker = ImagePicker();

  // 상태 변수들
  File? _originalImageFile;
  img.Image? _originalImage;
  img.Image? _resultImage;
  List<img.Image> _augmentedImages = [];

  bool _isProcessing = false;
  double _progress = 0.0;
  String _statusMessage = '';

  /// 갤러리에서 이미지 선택
  Future<void> _pickImage() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        maxHeight: 1920,
      );

      if (pickedFile != null) {
        setState(() {
          _originalImageFile = File(pickedFile.path);
          _resultImage = null;
          _augmentedImages = [];
          _statusMessage = '이미지를 선택했습니다.';
        });

        // 이미지 로드
        final image = await ImageProcessor.loadImage(pickedFile.path);
        if (image != null) {
          setState(() {
            _originalImage = image;
            _statusMessage = '이미지 로드 완료 (${image.width}x${image.height})';
          });
        }
      }
    } catch (e) {
      _showError('이미지 선택 실패: $e');
    }
  }

  /// 색감 보정 처리 시작
  Future<void> _processImage() async {
    if (_originalImage == null) {
      _showError('먼저 이미지를 선택해주세요.');
      return;
    }

    setState(() {
      _isProcessing = true;
      _progress = 0.0;
      _statusMessage = '이미지 증강 중...';
      _augmentedImages = [];
      _resultImage = null;
    });

    try {
      // 1단계: 증강된 이미지들 생성
      _augmentedImages = await ImageProcessor.generateAugmentedImages(
        _originalImage!,
        (progress) {
          setState(() {
            _progress = progress * 0.5; // 전체 진행의 50%
            _statusMessage = '이미지 증강 중... ${(_progress * 100).toInt()}%';
          });
        },
      );

      setState(() {
        _statusMessage = '${_augmentedImages.length}개 이미지 생성 완료. 합성 중...';
      });

      // 2단계: 최빈값 기반 이미지 합성
      final result = await ImageProcessor.synthesizeByMode(
        _augmentedImages,
        (progress) {
          setState(() {
            _progress = 0.5 + progress * 0.5; // 전체 진행의 50%~100%
            _statusMessage = '이미지 합성 중... ${(_progress * 100).toInt()}%';
          });
        },
      );

      setState(() {
        _resultImage = result;
        _isProcessing = false;
        _progress = 1.0;
        _statusMessage = '처리 완료!';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
      });
      _showError('이미지 처리 실패: $e');
    }
  }

  /// 결과 이미지 저장
  Future<void> _saveResultImage() async {
    if (_resultImage == null) {
      _showError('저장할 이미지가 없습니다.');
      return;
    }

    try {
      setState(() {
        _statusMessage = '이미지 저장 중...';
      });

      // 저장 경로 가져오기
      final directory = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final path = '${directory.path}/result_$timestamp.jpg';

      // 이미지 저장
      final success = await ImageProcessor.saveImage(_resultImage!, path);

      if (success) {
        _showSuccess('이미지가 저장되었습니다:\n$path');
      } else {
        _showError('이미지 저장 실패');
      }
    } catch (e) {
      _showError('이미지 저장 실패: $e');
    }
  }

  /// 에러 메시지 표시
  void _showError(String message) {
    setState(() {
      _statusMessage = message;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
      ),
    );
  }

  /// 성공 메시지 표시
  void _showSuccess(String message) {
    setState(() {
      _statusMessage = '저장 완료';
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green,
        duration: const Duration(seconds: 3),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('이미지 색감 보정 앱'),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 버튼 영역
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isProcessing ? null : _pickImage,
                      icon: const Icon(Icons.image),
                      label: const Text('이미지 선택'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.all(16),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: (_originalImage == null || _isProcessing)
                          ? null
                          : _processImage,
                      icon: const Icon(Icons.auto_fix_high),
                      label: const Text('색감 보정 처리'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.all(16),
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 16),

              // 진행 상황 표시
              if (_isProcessing) ...[
                LinearProgressIndicator(value: _progress),
                const SizedBox(height: 8),
              ],

              // 상태 메시지
              if (_statusMessage.isNotEmpty)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Text(
                      _statusMessage,
                      style: Theme.of(context).textTheme.bodyMedium,
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),

              const SizedBox(height: 16),

              // 원본 이미지와 결과 이미지 비교
              if (_originalImageFile != null)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '이미지 비교',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            // 원본 이미지
                            Expanded(
                              child: Column(
                                children: [
                                  const Text('원본'),
                                  const SizedBox(height: 8),
                                  Image.file(
                                    _originalImageFile!,
                                    fit: BoxFit.contain,
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(width: 16),
                            // 결과 이미지
                            Expanded(
                              child: Column(
                                children: [
                                  const Text('결과'),
                                  const SizedBox(height: 8),
                                  if (_resultImage != null)
                                    Image.memory(
                                      img.encodeJpg(_resultImage!),
                                      fit: BoxFit.contain,
                                    )
                                  else
                                    Container(
                                      height: 200,
                                      color: Colors.grey[300],
                                      child: const Center(
                                        child: Text('처리 대기 중'),
                                      ),
                                    ),
                                ],
                              ),
                            ),
                          ],
                        ),
                        if (_resultImage != null) ...[
                          const SizedBox(height: 16),
                          ElevatedButton.icon(
                            onPressed: _saveResultImage,
                            icon: const Icon(Icons.save),
                            label: const Text('결과 이미지 저장'),
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.all(16),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                ),

              const SizedBox(height: 16),

              // 증강된 이미지 그리드
              if (_augmentedImages.isNotEmpty)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '증강된 이미지 (총 ${_augmentedImages.length}개)',
                          style: Theme.of(context).textTheme.titleLarge,
                        ),
                        const SizedBox(height: 8),
                        Text(
                          '밝기(-20%, 0%, +20%) × 대비(0.8, 1.0, 1.2) × 채도(0.8, 1.0, 1.2)',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        const SizedBox(height: 16),
                        GridView.builder(
                          shrinkWrap: true,
                          physics: const NeverScrollableScrollPhysics(),
                          gridDelegate:
                              const SliverGridDelegateWithFixedCrossAxisCount(
                            crossAxisCount: 3,
                            crossAxisSpacing: 8,
                            mainAxisSpacing: 8,
                          ),
                          itemCount: _augmentedImages.length,
                          itemBuilder: (context, index) {
                            return Container(
                              decoration: BoxDecoration(
                                border: Border.all(color: Colors.grey),
                              ),
                              child: Image.memory(
                                img.encodeJpg(_augmentedImages[index]),
                                fit: BoxFit.cover,
                              ),
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
