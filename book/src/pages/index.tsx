import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className="hero">
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className="hero__buttons">
          <Link
            className="button button--primary button--lg"
            to="/part-1-foundations/ch-1-01">
            Start Reading
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/part-1-foundations/ch-1-01#learning-objectives"
            style={{marginLeft: '1rem'}}>
            View Curriculum
          </Link>
        </div>
      </div>
    </header>
  );
}

function FeatureCard({title, description, icon}: {title: string; description: string; icon: string}) {
  return (
    <div className="col col--4">
      <div className="card">
        <div className="card__header">
          <h3>{icon} {title}</h3>
        </div>
        <div className="card__body">
          <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

function HomepageFeatures() {
  const features = [
    {
      title: '21 Comprehensive Chapters',
      description: 'From ROS 2 fundamentals to advanced humanoid control, covering perception, planning, control, and machine learning.',
      icon: '📚',
    },
    {
      title: 'Simulation-First Approach',
      description: 'Learn with Gazebo, NVIDIA Isaac Sim, and MuJoCo. Practice safely before deploying to real hardware.',
      icon: '🤖',
    },
    {
      title: 'Hands-On Code Examples',
      description: 'Every concept includes working Python code examples you can run and modify in simulation.',
      icon: '💻',
    },
  ];

  return (
    <section className="features">
      <div className="container">
        <div className="row">
          {features.map((feature, idx) => (
            <FeatureCard key={idx} {...feature} />
          ))}
        </div>
      </div>
    </section>
  );
}

function TableOfContents() {
  const parts = [
    {title: 'Part 1: Foundations', chapters: ['Introduction', 'ROS 2 Fundamentals', 'Simulation Basics'], link: '/part-1-foundations/ch-1-01'},
    {title: 'Part 2: Perception', chapters: ['Computer Vision', 'Sensor Fusion', '3D Perception'], link: '/part-2-perception/ch-2-04'},
    {title: 'Part 3: Planning', chapters: ['Motion Planning', 'Task Planning', 'Behavior Trees'], link: '/part-3-planning/ch-3-07'},
    {title: 'Part 4: Control', chapters: ['PID Control', 'Force Control', 'Whole-Body Control'], link: '/part-4-control/ch-4-10'},
    {title: 'Part 5: Learning', chapters: ['Reinforcement Learning', 'Imitation Learning', 'VLA Models'], link: '/part-5-learning/ch-5-13'},
    {title: 'Part 6: Humanoids', chapters: ['Humanoid Kinematics', 'Bipedal Locomotion', 'Manipulation'], link: '/part-6-humanoids/ch-6-16'},
    {title: 'Part 7: Integration', chapters: ['System Integration', 'Safety Standards', 'Future Directions'], link: '/part-7-integration/ch-7-19'},
  ];

  return (
    <section className="toc-section">
      <div className="container">
        <h2>What You'll Learn</h2>
        <div className="row">
          {parts.map((part, idx) => (
            <div key={idx} className="col col--6" style={{marginBottom: '1rem'}}>
              <Link to={part.link} className="toc-card">
                <h4>{part.title}</h4>
                <ul>
                  {part.chapters.map((ch, i) => (
                    <li key={i}>{ch}</li>
                  ))}
                </ul>
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): React.JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Learn Physical AI and Humanoid Robotics from theory to simulation">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <TableOfContents />
      </main>
    </Layout>
  );
}
