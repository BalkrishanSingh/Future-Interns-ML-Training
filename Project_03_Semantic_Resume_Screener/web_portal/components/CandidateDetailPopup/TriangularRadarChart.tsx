'use client';

import React from 'react';

export interface RadarChartProps {
  skills: number;
  experience: number;
  education: number;
}

/**
 * Renders a crisp 3-axis SVG radar chart visualizing skills, experience, and education.
 */
export function TriangularRadarChart({ skills, experience, education }: RadarChartProps) {
  const width = 250;
  const height = 180;
  const centerX = width / 2;
  const centerY = height / 2 + 5;
  const maxRadius = 55;

  // Angles for 3 vertices (Skills at top, Experience at bottom-right, Education at bottom-left)
  const angleSkills = -Math.PI / 2;
  const angleExp = Math.PI / 6;
  const angleEdu = (5 * Math.PI) / 6;

  const clamp = (val: number) => Math.min(100, Math.max(0, val || 0)) / 100;
  const rSkills = clamp(skills) * maxRadius;
  const rExp = clamp(experience) * maxRadius;
  const rEdu = clamp(education) * maxRadius;

  const pSkills = {
    x: centerX + rSkills * Math.cos(angleSkills),
    y: centerY + rSkills * Math.sin(angleSkills),
  };
  const pExp = {
    x: centerX + rExp * Math.cos(angleExp),
    y: centerY + rExp * Math.sin(angleExp),
  };
  const pEdu = {
    x: centerX + rEdu * Math.cos(angleEdu),
    y: centerY + rEdu * Math.sin(angleEdu),
  };

  const polyPoints = `${pSkills.x},${pSkills.y} ${pExp.x},${pExp.y} ${pEdu.x},${pEdu.y}`;

  // Grid background polygons (at 33%, 66%, 100%)
  const gridLevels = [0.33, 0.66, 1.0];
  const gridPolys = gridLevels.map((lvl) => {
    const r = lvl * maxRadius;
    const p1 = `${centerX + r * Math.cos(angleSkills)},${centerY + r * Math.sin(angleSkills)}`;
    const p2 = `${centerX + r * Math.cos(angleExp)},${centerY + r * Math.sin(angleExp)}`;
    const p3 = `${centerX + r * Math.cos(angleEdu)},${centerY + r * Math.sin(angleEdu)}`;
    return `${p1} ${p2} ${p3}`;
  });

  return (
    <div className="relative flex flex-col items-center justify-center py-1 w-full max-w-full overflow-hidden">
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="max-w-full h-auto"
        aria-label="Candidate 3-axis radar chart"
      >
        {/* Background Grids */}
        {gridPolys.map((points, idx) => (
          <polygon
            key={idx}
            points={points}
            fill="none"
            stroke="#ccd0da"
            strokeWidth="1"
            strokeDasharray={idx < 2 ? '3,3' : undefined}
          />
        ))}

        {/* Axis Lines from Center */}
        <line
          x1={centerX}
          y1={centerY}
          x2={centerX + maxRadius * Math.cos(angleSkills)}
          y2={centerY + maxRadius * Math.sin(angleSkills)}
          stroke="#ccd0da"
          strokeWidth="1"
        />
        <line
          x1={centerX}
          y1={centerY}
          x2={centerX + maxRadius * Math.cos(angleExp)}
          y2={centerY + maxRadius * Math.sin(angleExp)}
          stroke="#ccd0da"
          strokeWidth="1"
        />
        <line
          x1={centerX}
          y1={centerY}
          x2={centerX + maxRadius * Math.cos(angleEdu)}
          y2={centerY + maxRadius * Math.sin(angleEdu)}
          stroke="#ccd0da"
          strokeWidth="1"
        />

        {/* Data Polygon */}
        <polygon
          points={polyPoints}
          fill="rgba(114, 135, 253, 0.25)"
          stroke="#7287fd"
          strokeWidth="2"
          className="transition-all duration-300 ease-out"
        />

        {/* Data Vertices */}
        <circle cx={pSkills.x} cy={pSkills.y} r="3.5" fill="#7287fd" />
        <circle cx={pExp.x} cy={pExp.y} r="3.5" fill="#7287fd" />
        <circle cx={pEdu.x} cy={pEdu.y} r="3.5" fill="#7287fd" />

        {/* Axis Labels */}
        <text
          x={centerX}
          y={centerY - maxRadius - 8}
          textAnchor="middle"
          className="text-[11px] font-semibold fill-[#4c4f69]"
        >
          Skills ({Math.round(skills)}%)
        </text>
        <text
          x={centerX + maxRadius * Math.cos(angleExp) + 6}
          y={centerY + maxRadius * Math.sin(angleExp) + 12}
          textAnchor="start"
          className="text-[11px] font-semibold fill-[#4c4f69]"
        >
          Exp ({Math.round(experience)}%)
        </text>
        <text
          x={centerX + maxRadius * Math.cos(angleEdu) - 6}
          y={centerY + maxRadius * Math.sin(angleEdu) + 12}
          textAnchor="end"
          className="text-[11px] font-semibold fill-[#4c4f69]"
        >
          Edu ({Math.round(education)}%)
        </text>
      </svg>
    </div>
  );
}

export default TriangularRadarChart;
