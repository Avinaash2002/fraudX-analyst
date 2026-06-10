/// FraudX Analyst - Models Screen
/// =================================

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_provider.dart';
import '../services/pdf_report_service.dart';
import '../services/tutorial_service.dart';

class ModelsScreen extends StatefulWidget {
  const ModelsScreen({super.key});
  @override
  State<ModelsScreen> createState() => _ModelsScreenState();
}

class _ModelsScreenState extends State<ModelsScreen> {
  int _activeTab = 1;

  final Map<String, Color> _modelColors = {
    'XGBoost': const Color(0xFF4CAF50),
    'LightGBM': const Color(0xFF2196F3),
    'Autoencoder': const Color(0xFFFF9800),
  };

  // Metric explanations for ℹ️ tooltips
  static const Map<String, String> _metricExplanations = {
    'Accuracy': 'The percentage of all predictions (fraud and normal) that the model got correct. High accuracy can be misleading with imbalanced datasets.',
    'Precision': 'Of all transactions flagged as fraud, how many were actually fraud? Higher precision = fewer false alarms.',
    'Recall': 'Of all actual fraud cases, how many did the model catch? Higher recall = fewer missed frauds.',
    'F1 Score': 'The harmonic mean of precision and recall. Balances both metrics, the best single measure for imbalanced fraud detection.',
    'AUC-ROC': 'Area Under the ROC Curve. Measures how well the model distinguishes between fraud and normal across all thresholds. 1.0 = perfect, 0.5 = random.',
  };

  void _showMetricExplanation(BuildContext context, String metric) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Row(children: [
          const Icon(Icons.info_outline, color: Color(0xFF2A9D8F), size: 22),
          const SizedBox(width: 10),
          Text(metric, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
        ]),
        content: Text(_metricExplanations[metric] ?? '', style: const TextStyle(fontSize: 14, height: 1.5, color: Color(0xFF374151))),
        actions: [TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Got it', style: TextStyle(color: Color(0xFF2A9D8F), fontWeight: FontWeight.w600)))],
      ),
    ).then((_) {
      try {
        final tutorial = context.read<TutorialService>();
        if (tutorial.isActive && tutorial.waitingForAction) {
          tutorial.completeAction();
        }
      } catch (_) {}
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      body: SafeArea(
        child: Consumer<AppProvider>(builder: (context, provider, _) {
          final bestModel = provider.bestModel;
          return SingleChildScrollView(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            // ── Header ──────────────────────────────────────────
            Padding(padding: const EdgeInsets.fromLTRB(20, 16, 20, 0), child: Row(children: [
              Container(width: 36, height: 36, decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 6)]),
                child: const Icon(Icons.arrow_back_ios_new, size: 16, color: Color(0xFF1A1A2E))),
              const SizedBox(width: 16),
              const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('ML Models', style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800, color: Color(0xFF1A1A2E))),
                Text('Compare model performance', style: TextStyle(fontSize: 14, color: Color(0xFF6B7280))),
              ]),
            ])),
            const SizedBox(height: 20),

            // ── MLFlow Registry Card ────────────────────────────
            Padding(padding: const EdgeInsets.symmetric(horizontal: 20), child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
              decoration: BoxDecoration(color: const Color(0xFFE0F2F1), borderRadius: BorderRadius.circular(16)),
              child: Row(children: [
                Container(width: 40, height: 40, decoration: BoxDecoration(color: const Color(0xFF2A9D8F), borderRadius: BorderRadius.circular(12)),
                  child: const Icon(Icons.hub, size: 20, color: Colors.white)),
                const SizedBox(width: 14),
                Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  const Text('MLFlow Registry', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
                  Text('${provider.models.isNotEmpty ? provider.models.length : 3} models available', style: const TextStyle(fontSize: 13, color: Color(0xFF6B7280))),
                ])),
                GestureDetector(onTap: () => provider.loadModels(), child: const Icon(Icons.refresh, size: 22, color: Color(0xFF2A9D8F))),
              ]),
            )),
            const SizedBox(height: 18),

            // ── Tabs ────────────────────────────────────────────
            Padding(padding: const EdgeInsets.symmetric(horizontal: 20), child: Row(children: [
              _Tab(label: 'Models', isActive: _activeTab == 0, onTap: () => setState(() => _activeTab = 0)),
              const SizedBox(width: 10),
              _Tab(label: 'Comparison', isActive: _activeTab == 1, onTap: () => setState(() => _activeTab = 1)),
            ])),
            const SizedBox(height: 14),

            if (_activeTab == 1)
              Padding(padding: const EdgeInsets.symmetric(horizontal: 20), child: Center(child: OutlinedButton.icon(
                onPressed: () async {
                  try {
                    final models = provider.models;
                    if (models.isEmpty) return;
                    final best = bestModel?.modelName ?? models.first.modelName;
                    final data = models.map((m) => <String, dynamic>{
                      'name': m.modelName,
                      'accuracy': m.accuracy,
                      'precision': m.precision,
                      'recall': m.recall,
                      'f1': m.f1Score,
                      'auc': m.aucRoc,
                      'time': m.trainingTime ?? 0.0,
                    }).toList();
                    await PdfReportService.generateComparisonReport(models: data, bestModelName: best);
                    if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('PDF report generated!'), backgroundColor: Color(0xFF2A9D8F)));
                  } catch (e) {
                    if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('PDF error: $e'), backgroundColor: const Color(0xFFEF4444)));
                  }
                },
                icon: const Icon(Icons.picture_as_pdf, size: 18),
                label: const Text('Download PDF', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
                style: OutlinedButton.styleFrom(
                  foregroundColor: const Color(0xFF2A9D8F),
                  side: const BorderSide(color: Color(0xFF2A9D8F)),
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
              ))),
            const SizedBox(height: 16),

            if (_activeTab == 0) _buildModelsList(provider, bestModel),
            if (_activeTab == 1) _buildComparison(provider, bestModel),
          ]));
        }),
      ),
    );
  }

  Widget _buildModelsList(AppProvider provider, dynamic bestModel) {
    if (provider.models.isEmpty) return const Padding(padding: EdgeInsets.all(40), child: Center(child: Text('No models loaded yet', style: TextStyle(color: Color(0xFF6B7280)))));
    return Padding(padding: const EdgeInsets.symmetric(horizontal: 20), child: Column(children: provider.models.map((model) {
      final color = _modelColors[model.modelName] ?? const Color(0xFF6B7280);
      final isBest = bestModel != null && model.modelName == bestModel.modelName;
      return Container(
        margin: const EdgeInsets.only(bottom: 12), padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16),
          border: isBest ? Border.all(color: const Color(0xFF2A9D8F), width: 2) : null,
          boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Container(width: 12, height: 12, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
            const SizedBox(width: 10),
            Text(model.modelName, style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
            if (isBest) ...[
              const SizedBox(width: 8),
              Container(padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3), decoration: BoxDecoration(color: const Color(0xFFE0F2F1), borderRadius: BorderRadius.circular(8)),
                child: const Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.emoji_events, size: 12, color: Color(0xFF2A9D8F)),
                  SizedBox(width: 4),
                  Text('BEST', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: Color(0xFF2A9D8F))),
                ])),
            ],
            const Spacer(),
            Container(padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4), decoration: BoxDecoration(color: const Color(0xFFF3F4F6), borderRadius: BorderRadius.circular(10)),
              child: Text(model.algorithmType, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Color(0xFF6B7280)))),
          ]),
          if (model.description != null) ...[const SizedBox(height: 8), Text(model.description!, style: const TextStyle(fontSize: 13, color: Color(0xFF6B7280)))],
          const SizedBox(height: 14),
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            _SmallMetric(label: 'Accuracy', value: '${(model.accuracy * 100).toStringAsFixed(2)}%'),
            _SmallMetric(label: 'Precision', value: '${(model.precision * 100).toStringAsFixed(2)}%'),
            _SmallMetric(label: 'Recall', value: '${(model.recall * 100).toStringAsFixed(2)}%'),
            _SmallMetric(label: 'F1', value: '${(model.f1Score * 100).toStringAsFixed(2)}%'),
          ]),
        ]),
      );
    }).toList()));
  }

  Widget _buildComparison(AppProvider provider, dynamic bestModel) {
    final models = provider.models;
    if (models.isEmpty) return const Padding(padding: EdgeInsets.all(40), child: Center(child: Text('Loading model data…', style: TextStyle(color: Color(0xFF6B7280)))));
    final metrics = [
      {'label': 'Accuracy', 'getter': (m) => m.accuracy},
      {'label': 'Precision', 'getter': (m) => m.precision},
      {'label': 'Recall', 'getter': (m) => m.recall},
      {'label': 'F1 Score', 'getter': (m) => m.f1Score},
      {'label': 'AUC-ROC', 'getter': (m) => m.aucRoc},
    ];
    return Padding(padding: const EdgeInsets.symmetric(horizontal: 20), child: Column(children: [
      ...metrics.map((metric) {
      final label = metric['label'] as String;
      final getter = metric['getter'] as Function;
      return Container(
        margin: const EdgeInsets.only(bottom: 14), padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 8, offset: const Offset(0, 2))]),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            const Icon(Icons.show_chart, size: 18, color: Color(0xFF2A9D8F)),
            const SizedBox(width: 8),
            Text(label, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E))),
            const Spacer(),
            Consumer<TutorialService>(
              builder: (context, tutorial, _) {
                final isGlowing = tutorial.isActive && tutorial.currentStep == 12 && tutorial.waitingForAction;
                if (isGlowing) {
                  return _BlinkingIcon(onTap: () => _showMetricExplanation(context, label));
                }
                return GestureDetector(
                  onTap: () => _showMetricExplanation(context, label),
                  child: const Icon(Icons.info_outline, size: 18, color: Color(0xFF9CA3AF)),
                );
              },
            ),
          ]),
          const SizedBox(height: 14),
          ...models.map((model) {
            final value = getter(model) as double;
            final color = _modelColors[model.modelName] ?? const Color(0xFF6B7280);
            final isBest = bestModel != null && model.modelName == bestModel.modelName;
            return Padding(padding: const EdgeInsets.only(bottom: 10), child: Row(children: [
              SizedBox(width: 80, child: Row(children: [
                Text(model.modelName.length > 10 ? '${model.modelName.substring(0, 10)}…' : model.modelName, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500, color: Color(0xFF6B7280))),
                if (isBest) const Padding(padding: EdgeInsets.only(left: 4), child: Icon(Icons.emoji_events, size: 12, color: Color(0xFF2A9D8F))),
              ])),
              const SizedBox(width: 8),
              Expanded(child: ClipRRect(borderRadius: BorderRadius.circular(6),
                child: LinearProgressIndicator(value: value, minHeight: 14, backgroundColor: const Color(0xFFF3F4F6), color: color))),
              const SizedBox(width: 10),
              SizedBox(width: 50, child: Text('${(value * 100).toStringAsFixed(2)}%', textAlign: TextAlign.end, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: Color(0xFF1A1A2E)))),
            ]));
          }),
        ]),
      );
    }),

      // ── Best Performing Model Card ────────────────────────
      if (bestModel != null) Container(
        margin: const EdgeInsets.only(bottom: 20), padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: const LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [Color(0xFF2A9D8F), Color(0xFF38BDF8), Color(0xFF6BCB77)]),
          borderRadius: BorderRadius.circular(20),
          boxShadow: [BoxShadow(color: const Color(0xFF2A9D8F).withOpacity(0.3), blurRadius: 12, offset: const Offset(0, 4))],
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            const Text('🏆', style: TextStyle(fontSize: 20)),
            const SizedBox(width: 8),
            const Text('Best Performing Model', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800, color: Colors.white)),
          ]),
          const SizedBox(height: 10),
          Text(
            '${bestModel.modelName} achieves the highest overall performance with ${(bestModel.accuracy * 100).toStringAsFixed(2)}% accuracy, making it the recommended model for production deployment.',
            style: TextStyle(fontSize: 14, height: 1.5, color: Colors.white.withOpacity(0.9)),
          ),
        ]),
      ),
    ]));
  }
}

class _Tab extends StatelessWidget {
  final String label; final bool isActive; final VoidCallback onTap;
  const _Tab({required this.label, required this.isActive, required this.onTap});
  @override
  Widget build(BuildContext context) {
    return GestureDetector(onTap: onTap, child: Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      decoration: BoxDecoration(color: isActive ? Colors.white : Colors.transparent, borderRadius: BorderRadius.circular(12),
        boxShadow: isActive ? [BoxShadow(color: Colors.black.withOpacity(0.06), blurRadius: 6, offset: const Offset(0, 2))] : null),
      child: Text(label, style: TextStyle(fontSize: 14, fontWeight: isActive ? FontWeight.w600 : FontWeight.w400, color: isActive ? const Color(0xFF1A1A2E) : const Color(0xFF6B7280))),
    ));
  }
}

class _SmallMetric extends StatelessWidget {
  final String label, value;
  const _SmallMetric({required this.label, required this.value});
  @override
  Widget build(BuildContext context) {
    return Column(children: [
      Text(value, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Color(0xFF2A9D8F))),
      const SizedBox(height: 2),
      Text(label, style: const TextStyle(fontSize: 11, color: Color(0xFF6B7280))),
    ]);
  }
}

class _BlinkingIcon extends StatefulWidget {
  final VoidCallback onTap;
  const _BlinkingIcon({required this.onTap});
  @override
  State<_BlinkingIcon> createState() => _BlinkingIconState();
}

class _BlinkingIconState extends State<_BlinkingIcon> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 600))..repeat(reverse: true);
  }

  @override
  void dispose() { _controller.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        final color = Color.lerp(const Color(0xFF2A9D8F), const Color(0xFF38BDF8), _controller.value)!;
        return GestureDetector(
          onTap: widget.onTap,
          child: Container(
            padding: const EdgeInsets.all(6),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color.withOpacity(0.15 + _controller.value * 0.15),
            ),
            child: Icon(Icons.info_outline, size: 20, color: color),
          ),
        );
      },
    );
  }
}
