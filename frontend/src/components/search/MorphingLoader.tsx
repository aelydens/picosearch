"use client";

import { motion } from "framer-motion";

const BLOBS_A = [
  "60% 40% 30% 70% / 60% 30% 70% 40%",
  "40% 60% 70% 30% / 40% 70% 30% 60%",
  "70% 30% 50% 50% / 30% 50% 50% 70%",
  "30% 70% 40% 60% / 70% 40% 60% 30%",
  "60% 40% 30% 70% / 60% 30% 70% 40%",
];

const BLOBS_B = [
  "30% 70% 40% 60% / 70% 40% 60% 30%",
  "70% 30% 50% 50% / 30% 50% 50% 70%",
  "40% 60% 70% 30% / 40% 70% 30% 60%",
  "60% 40% 30% 70% / 60% 30% 70% 40%",
  "30% 70% 40% 60% / 70% 40% 60% 30%",
];

interface Props {
  label?: string;
  size?: "sm" | "md" | "lg";
}

const SIZES = {
  sm: { container: "h-12 w-12", inner: "inset-1.5", dot: "inset-[38%]" },
  md: { container: "h-20 w-20", inner: "inset-2", dot: "inset-[36%]" },
  lg: { container: "h-28 w-28", inner: "inset-3", dot: "inset-[35%]" },
};

export function MorphingLoader({ label = "Searching…", size = "lg" }: Props) {
  const s = SIZES[size];

  return (
    <div className="flex flex-col items-center gap-5 py-10">
      <div className={`relative ${s.container}`}>
        {/* Amber outer blob */}
        <motion.div
          className="absolute inset-0 bg-amber-400"
          animate={{ borderRadius: BLOBS_A, rotate: [0, 120, 240, 360], scale: [1, 1.08, 0.96, 1.04, 1] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        />
        {/* Violet inner blob */}
        <motion.div
          className={`absolute ${s.inner} bg-violet-600`}
          animate={{ borderRadius: BLOBS_B, rotate: [0, -90, -180, -270, -360], scale: [1, 0.92, 1.08, 0.97, 1] }}
          transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut", delay: 0.3 }}
        />
        {/* White core */}
        <motion.div
          className={`absolute ${s.dot} rounded-full bg-white`}
          animate={{ scale: [1, 1.25, 1], opacity: [0.85, 1, 0.85] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      {label && (
        <motion.p
          className="text-sm font-medium text-violet-500"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
        >
          {label}
        </motion.p>
      )}
    </div>
  );
}
