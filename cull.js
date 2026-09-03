#!/usr/bin/env node
'use strict';
const fs = require('fs');
const path = require('path');
const exifr = require('exifr');
const { analyzeImage } = require('./lib/vision');

const IMAGE_EXT = new Set(['.jpg', '.jpeg', '.png', '.heic', '.heif', '.tif', '.tiff']);

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next === undefined || next.startsWith('--')) {
        args[key] = true;
      } else {
        args[key] = next;
        i++;
      }
    } else {
      args._.push(a);
    }
  }
  return args;
}

async function listImages(dir) {
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  return entries
    .filter((e) => e.isFile() && IMAGE_EXT.has(path.extname(e.name).toLowerCase()))
    .map((e) => path.join(dir, e.name))
    .sort();
}

async function getCaptureTime(filePath) {
  try {
    const exif = await exifr.parse(filePath, ['DateTimeOriginal', 'CreateDate']);
    const t = exif && (exif.DateTimeOriginal || exif.CreateDate);
    if (t instanceof Date && !isNaN(t)) return t.getTime();
  } catch (_) {
    // EXIF 없음 또는 파싱 실패 -> 파일 수정시간으로 대체
  }
  const stat = await fs.promises.stat(filePath);
  return stat.mtimeMs;
}

// 촬영시각 간격이 gapSeconds 이내면 같은 연사 그룹으로 묶음
function groupByBurst(items, gapSeconds) {
  const sorted = [...items].sort((a, b) => a.time - b.time);
  const groups = [];
  let current = [];
  for (const item of sorted) {
    if (current.length === 0) {
      current.push(item);
      continue;
    }
    const prev = current[current.length - 1];
    const gap = (item.time - prev.time) / 1000;
    if (gap <= gapSeconds) {
      current.push(item);
    } else {
      groups.push(current);
      current = [item];
    }
  }
  if (current.length) groups.push(current);
  return groups;
}

function scoreImage(analysis, maxSharpnessInGroup) {
  const sharpNorm = maxSharpnessInGroup > 0
    ? Math.min(1, analysis.sharpness / maxSharpnessInGroup)
    : 1;

  let eyeScore = 1;
  let exprScore = 0.5;
  let hardPenalty = false;
  let reason = [];

  if (analysis.faces.length > 0) {
    const totalArea = analysis.faces.reduce((s, f) => s + f.areaFrac, 0) || 1;
    eyeScore = analysis.faces.reduce((s, f) => s + f.openness * (f.areaFrac / totalArea), 0);
    const exprRaw = analysis.faces.reduce((s, f) => s + f.exprScoreRaw * (f.areaFrac / totalArea), 0);
    exprScore = Math.min(1, Math.max(0, (exprRaw + 1) / 2));

    // 얼굴 면적 비중이 큰(주요 인물) 얼굴이 눈을 감았으면 강한 페널티
    const mainFace = [...analysis.faces].sort((a, b) => b.areaFrac - a.areaFrac)[0];
    if (mainFace.eyesClosed) {
      hardPenalty = true;
      reason.push('주요 인물 눈 감음');
    }
  }

  if (sharpNorm < 0.15) {
    hardPenalty = true;
    reason.push('심한 블러');
  }

  let composite = 0.35 * sharpNorm + 0.4 * eyeScore + 0.25 * exprScore;
  if (hardPenalty) composite *= 0.3;

  return { composite, sharpNorm, eyeScore, exprScore, hardPenalty, reason };
}

function selectKeepers(scoredGroup, closeness) {
  const sorted = [...scoredGroup].sort((a, b) => b.score.composite - a.score.composite);
  if (sorted.length === 0) return [];
  if (sorted.length === 1) {
    sorted[0].selected = true;
    sorted[0].selectReason = sorted.length === 1 ? '단독 컷' : '1위';
    return sorted;
  }
  sorted[0].selected = true;
  sorted[0].selectReason = '1위';
  const second = sorted[1];
  const top = sorted[0].score.composite;
  if (
    top > 0 &&
    second.score.composite / top >= closeness &&
    !second.score.hardPenalty &&
    second.score.sharpNorm > 0.4
  ) {
    second.selected = true;
    second.selectReason = `2위 (1위와 점수차 ${(100 - (second.score.composite / top) * 100).toFixed(1)}% 이내)`;
  }
  return sorted;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputDir = args._[0];
  if (!inputDir || args.help || args.h) {
    console.log(`사용법: node cull.js <입력폴더> --out <출력폴더> [옵션]

옵션:
  --out <dir>          선별된 사진을 복사할 폴더 (기본: ./selected)
  --gap <seconds>       같은 연사로 묶을 촬영 간격(초) (기본: 1.5)
  --closeness <0-1>     1위 대비 이 비율 이상 점수면 2위도 함께 선택 (기본: 0.9)
  --dry-run             복사 없이 리포트만 생성

지원 포맷: JPEG, PNG, HEIC/HEIF, TIFF (RAW 원본은 미지원)`);
    process.exit(inputDir ? 0 : 1);
  }

  const outDir = path.resolve(args.out || './selected');
  const gapSeconds = parseFloat(args.gap || '1.5');
  const closeness = parseFloat(args.closeness || '0.9');
  const dryRun = !!args['dry-run'];

  const files = await listImages(path.resolve(inputDir));
  if (files.length === 0) {
    console.log('이미지 파일을 찾지 못했습니다:', inputDir);
    return;
  }
  console.log(`${files.length}장 발견. 촬영시각 확인 중...`);

  const items = [];
  for (const f of files) {
    const time = await getCaptureTime(f);
    items.push({ file: f, time });
  }

  const groups = groupByBurst(items, gapSeconds);
  console.log(`${groups.length}개 그룹으로 분류 (간격 기준 ${gapSeconds}초). 분석 시작...`);

  if (!dryRun) await fs.promises.mkdir(outDir, { recursive: true });

  const report = [];
  let selectedCount = 0;

  for (let gi = 0; gi < groups.length; gi++) {
    const group = groups[gi];
    const analyzed = [];
    for (const item of group) {
      process.stdout.write(`  [그룹 ${gi + 1}/${groups.length}] ${path.basename(item.file)} 분석 중...\r`);
      const analysis = await analyzeImage(item.file);
      analyzed.push({ ...item, analysis });
    }
    const maxSharpness = Math.max(...analyzed.map((a) => a.analysis.sharpness));
    const scored = analyzed.map((a) => ({
      ...a,
      score: scoreImage(a.analysis, maxSharpness),
    }));

    const ranked = selectKeepers(scored, closeness);

    for (let rank = 0; rank < ranked.length; rank++) {
      const r = ranked[rank];
      const baseName = path.basename(r.file);
      const groupTag = String(gi + 1).padStart(3, '0');
      const status = r.selected ? '선택' : '탈락';
      if (r.selected) {
        selectedCount++;
        if (!dryRun) {
          const destName = `${groupTag}-${rank + 1}_${baseName}`;
          await fs.promises.copyFile(r.file, path.join(outDir, destName));
        }
      }
      report.push({
        group: groupTag,
        file: baseName,
        faces: r.analysis.faces.length,
        sharpness: Math.round(r.analysis.sharpness),
        eyeScore: r.score.eyeScore.toFixed(2),
        exprScore: r.score.exprScore.toFixed(2),
        composite: r.score.composite.toFixed(3),
        status,
        reason: r.selected ? r.selectReason : r.score.reason.join(', '),
      });
    }
    process.stdout.write(' '.repeat(80) + '\r');
  }

  const csvLines = [
    'group,file,faces,sharpness,eyeScore,exprScore,composite,status,reason',
    ...report.map((r) =>
      [r.group, r.file, r.faces, r.sharpness, r.eyeScore, r.exprScore, r.composite, r.status, r.reason]
        .map((v) => `"${String(v).replace(/"/g, '""')}"`)
        .join(',')
    ),
  ];
  const reportPath = path.join(dryRun ? path.resolve('.') : outDir, 'report.csv');
  await fs.promises.writeFile(reportPath, csvLines.join('\n'), 'utf8');

  console.log(`완료: 총 ${files.length}장 중 ${selectedCount}장 선택 (${groups.length}개 그룹)`);
  console.log(`리포트: ${reportPath}`);
  if (!dryRun) console.log(`선택된 사진: ${outDir}`);
}

main().catch((e) => {
  console.error('오류:', e);
  process.exit(1);
});
