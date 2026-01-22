import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  bookSidebar: [
    {
      type: 'category',
      label: 'Part 1: Foundations',
      collapsed: false,
      items: [
        'part-1-foundations/ch-1-01',
        'part-1-foundations/ch-1-02',
        'part-1-foundations/ch-1-03',
      ],
    },
    {
      type: 'category',
      label: 'Part 2: Perception',
      items: [
        'part-2-perception/ch-2-04',
        'part-2-perception/ch-2-05',
        'part-2-perception/ch-2-06',
      ],
    },
    {
      type: 'category',
      label: 'Part 3: Planning',
      items: [
        'part-3-planning/ch-3-07',
        'part-3-planning/ch-3-08',
        'part-3-planning/ch-3-09',
      ],
    },
    {
      type: 'category',
      label: 'Part 4: Control',
      items: [
        'part-4-control/ch-4-10',
        'part-4-control/ch-4-11',
        'part-4-control/ch-4-12',
      ],
    },
    {
      type: 'category',
      label: 'Part 5: Learning',
      items: [
        'part-5-learning/ch-5-13',
        'part-5-learning/ch-5-14',
        'part-5-learning/ch-5-15',
      ],
    },
    {
      type: 'category',
      label: 'Part 6: Humanoids',
      items: [
        'part-6-humanoids/ch-6-16',
        'part-6-humanoids/ch-6-17',
        'part-6-humanoids/ch-6-18',
      ],
    },
    {
      type: 'category',
      label: 'Part 7: Integration',
      items: [
        'part-7-integration/ch-7-19',
        'part-7-integration/ch-7-20',
        'part-7-integration/ch-7-21',
      ],
    },
  ],
};

export default sidebars;
