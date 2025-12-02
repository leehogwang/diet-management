import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import '../services/image_processor.dart';

class ColorAugmentationPage extends StatefulWidget {
  const ColorAugmentationPage({super.key});

  @override
  State<ColorAugmentationPage> createState() => _ColorAugmentationPageState();
}

class _ColorAugmentationPageState extends State<ColorAugmentationPage> {
  final ImagePicker _picker = ImagePicker();

  Uint8List? _originalImageBytes;
  img.Image? _originalImage;
  img.Image? _resultImage;
  List<img.Image> _augmentedImages = [];

  bool _isProcessing = false;
  double _progress = 0.0;
  String _statusMessage = '';

  Future<void> _pickImage() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        maxHeight: 1920,
      );

      if (pickedFile != null) {
        final bytes = await pickedFile.readAsBytes();
        final image = img.decodeImage(bytes);

        setState(() {
          _originalImageBytes = bytes;
          _originalImage = image;
          _resultImage = null;
          _augmentedImages = [];
          _progress = 0.0;
          _statusMessage = '이미지가 선택되었습니다. 처리 버튼을 눌러주세요.';
        });
      }
    } catch (e) {
      _showError('이미지 선택 실패: $e');
    }
  }

  Future<void> _processImage() async {
    if (_originalImage == null) return;

    setState(() {
      _isProcessing = true;
      _progress = 0.0;
      _statusMessage = '증강 이미지 생성 중...';
      _augmentedImages = [];
      _resultImage = null;
    });

    try {
      // 1. 증강 이미지 생성 (50%)
      final augmented = await ImageProcessor.generateAugmentedImages(
        _originalImage!,
        (progress) {
          setState(() {
            _progress = progress * 0.5;
          });
        },
      );

      setState(() {
        _augmentedImages = augmented;
        _statusMessage = '최빈값 기반 합성 중...';
        _progress = 0.5;
      });

      // 2. 최빈값 합성 (50%)
      // UI 스레드 차단을 방지하기 위해 잠시 대기
      await Future.delayed(const Duration(milliseconds: 100));

      final result = await ImageProcessor.synthesizeByMode(
        augmented,
        (progress) {
          setState(() {
            _progress = 0.5 + (progress * 0.5);
          });
        },
      );

      setState(() {
        _resultImage = result;
        _isProcessing = false;
        _statusMessage = '완료!';
        _progress = 1.0;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
      });
      _showError('처리 중 오류 발생: $e');
    }
  }

  Future<void> _saveResult() async {
    if (_resultImage == null) return;

    try {
      final directory = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final path = '${directory.path}/augmented_result_$timestamp.jpg';

      final success = await ImageProcessor.saveImage(_resultImage!, path);

      if (success) {
        _showSuccess('저장 완료: $path');
      } else {
        _showError('저장 실패');
      }
    } catch (e) {
      _showError('저장 중 오류 발생: $e');
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  void _showSuccess(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.green),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('이미지 색감 보정'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 이미지 선택 버튼
            ElevatedButton.icon(
              onPressed: _isProcessing ? null : _pickImage,
              icon: const Icon(Icons.photo_library),
              label: const Text('갤러리에서 이미지 선택'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.all(16),
                textStyle: const TextStyle(fontSize: 18),
              ),
            ),
            const SizedBox(height: 16),

            // 원본 및 결과 이미지 비교
            if (_originalImageBytes != null)
              Row(
                children: [
                  Expanded(
                    child: Column(
                      children: [
                        const Text('원본', style: TextStyle(fontWeight: FontWeight.bold)),
                        const SizedBox(height: 8),
                        Image.memory(
                          _originalImageBytes!,
                          height: 200,
                          fit: BoxFit.contain,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      children: [
                        const Text('결과', style: TextStyle(fontWeight: FontWeight.bold)),
                        const SizedBox(height: 8),
                        if (_resultImage != null)
                          Image.memory(
                            img.encodeJpg(_resultImage!),
                            height: 200,
                            fit: BoxFit.contain,
                          )
                        else
                          Container(
                            height: 200,
                            color: Colors.grey[200],
                            child: const Center(child: Text('대기 중')),
                          ),
                      ],
                    ),
                  ),
                ],
              ),

            const SizedBox(height: 16),

            // 처리 버튼 및 진행률
            if (_originalImageBytes != null) ...[
              ElevatedButton.icon(
                onPressed: _isProcessing ? null : _processImage,
                icon: const Icon(Icons.auto_fix_high),
                label: const Text('색감 보정 처리'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Theme.of(context).colorScheme.primary,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.all(16),
                ),
              ),
              const SizedBox(height: 8),
              if (_isProcessing || _progress > 0) ...[
                LinearProgressIndicator(value: _progress),
                const SizedBox(height: 8),
                Text(_statusMessage, textAlign: TextAlign.center),
              ],
            ],

            const SizedBox(height: 16),

            // 저장 버튼
            if (_resultImage != null)
              ElevatedButton.icon(
                onPressed: _saveResult,
                icon: const Icon(Icons.save),
                label: const Text('결과 이미지 저장'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.all(16),
                ),
              ),

            const SizedBox(height: 24),

            // 증강 이미지 그리드
            if (_augmentedImages.isNotEmpty) ...[
              const Text(
                '증강된 이미지 (27개)',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              GridView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 3,
                  crossAxisSpacing: 4,
                  mainAxisSpacing: 4,
                ),
                itemCount: _augmentedImages.length,
                itemBuilder: (context, index) {
                  return Image.memory(
                    img.encodeJpg(_augmentedImages[index]), // 성능상 좋지 않지만 미리보기용
                    fit: BoxFit.cover,
                  );
                },
              ),
            ],
          ],
        ),
      ),
    );
  }
}
