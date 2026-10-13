const cStd = ['com', 'net', 'org', 'gov', 'edu'] as const;

const cStdMil = [...cStd, 'mil'] as const;

const dCcTLDPrefixes: Record<string, readonly string[]> = {
  // --- Standard 5 only ---
  ar: cStd,
  bd: cStd,
  bg: cStd,
  cn: cStdMil,
  eg: cStd,
  gr: cStd,
  hk: cStd,
  hr: cStd,
  lk: cStd,
  mx: cStdMil,
  my: cStdMil,
  ng: cStd,
  ph: cStd,
  pk: cStd,
  pl: cStd,
  ro: cStd,
  ru: cStd,
  sa: cStd,
  si: cStd,
  tr: cStd,
  tw: cStd,
  ua: cStd,
  ve: cStd,

  au: [...cStdMil, 'id', 'asn', 'csiro'],
  br: [
    ...cStdMil,
    'art',
    'eco',
    'eng',
    'inf',
    'med',
    'psi',
    'tmp',
    'etc',
    'adm',
    'adv',
    'arq',
    'bio',
    'bmd',
    'cim',
    'cng',
    'cnt',
    'coop',
    'ecn',
    'esp',
    'far',
    'fm',
    'fnd',
    'fot',
    'fst',
    'g12',
    'ggf',
    'imb',
    'ind',
    'jor',
    'jus',
    'leg',
    'lel',
    'mat',
    'mp',
    'mus',
    'not',
    'ntr',
    'odo',
    'ppg',
    'pro',
    'psc',
    'qsl',
    'rec',
    'slg',
    'srv',
    'trd',
    'tur',
    'tv',
    'vet',
    'vlog',
    'wiki',
    'zlg'
  ],
  id: [...cStdMil, 'co', 'go', 'or', 'web', 'sch'],
  in: [...cStdMil, 'co', 'gen', 'ind', 'firm', 'ernet', 'nic'],
  kr: [...cStdMil, 'co', 'go', 'or', 'ac', 're'],
  nz: [
    ...cStdMil,
    'co',
    'gen',
    'geek',
    'kiwi',
    'maori',
    'school',
    'govt',
    'health',
    'iwi',
    'parliament'
  ],
  sg: [...cStd, 'per'],
  th: ['co', 'go', 'or', 'in', 'ac', 'mi', 'net'],

  ae: ['co', 'net', 'org', 'gov', 'ac', 'sch'],
  hu: ['co', 'net', 'org', 'gov', 'edu'],
  il: ['co', 'net', 'org', 'gov', 'ac', 'muni'],
  jp: ['ac', 'ad', 'co', 'ed', 'go', 'gr', 'lg', 'ne', 'or'],
  ke: ['co', 'or', 'ne', 'go', 'ac', 'sc'],
  rs: ['co', 'net', 'org', 'gov', 'edu'],
  uk: ['co', 'org', 'net', 'ac', 'gov', 'mil', 'nhs', 'police', 'mod', 'ltd', 'plc', 'me', 'sch'],
  za: ['co', 'org', 'net', 'web', 'law', 'mil']
};

const dWildcardBases: Record<string, readonly string[]> = {
  br: ['nom', 'blog'],
  jp: [
    'kobe',
    'kyoto',
    'nagoya',
    'osaka',
    'sapporo',
    'sendai',
    'tokyo',
    'yokohama',
    'aichi',
    'akita',
    'aomori',
    'chiba',
    'ehime',
    'fukui',
    'fukuoka',
    'fukushima',
    'gifu',
    'gunma',
    'hiroshima',
    'hokkaido',
    'hyogo',
    'ibaraki',
    'ishikawa',
    'iwate',
    'kagawa',
    'kagoshima',
    'kanagawa',
    'kochi',
    'kumamoto',
    'mie',
    'miyagi',
    'miyazaki',
    'nagano',
    'nara',
    'niigata',
    'oita',
    'okayama',
    'okinawa',
    'saga',
    'saitama',
    'shiga',
    'shimane',
    'shizuoka',
    'tochigi',
    'tokushima',
    'tottori',
    'toyama',
    'wakayama',
    'yamagata',
    'yamaguchi',
    'yamanashi'
  ]
};

function fBuildSuffixSet(pSuffixes: Record<string, readonly string[]>): Set<string> {
  const sSet = new Set<string>();

  for (const [cTld, cParts] of Object.entries(pSuffixes)) {
    for (const cPart of cParts) {
      sSet.add(`${cPart}.${cTld}`);
    }
  }

  return sSet;
}

export const cTwoPartPublicSuffixes = fBuildSuffixSet(dCcTLDPrefixes);
export const cWildcardPublicSuffixes = fBuildSuffixSet(dWildcardBases);

// Matches one or more trailing "/" characters at the end of a URL/path.
export const cTrailingSlashesRegex = /\/+$/;
