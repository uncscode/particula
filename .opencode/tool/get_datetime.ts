/**
 * ADW Date/Time Tool
 *
 * Returns the current date or datetime in various formats (UTC or America/Denver).
 */

import { tool } from "@opencode-ai/plugin";

type DateTimeFormat = "date" | "datetime" | "human";

const buildOffset = (date: Date, timeZone: string): string => {
  const zoned = new Date(date.toLocaleString("en-US", { timeZone }));
  const offsetMinutes = zoned.getTimezoneOffset();
  const sign = offsetMinutes <= 0 ? "+" : "-";
  const absMinutes = Math.abs(offsetMinutes);
  const hours = String(Math.floor(absMinutes / 60)).padStart(2, "0");
  const minutes = String(absMinutes % 60).padStart(2, "0");

  return `${sign}${hours}:${minutes}`;
};

const formatParts = (date: Date, timeZone: string, includeSeconds = false) => {
  const options: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: includeSeconds ? "2-digit" : undefined,
    timeZone,
    hour12: false,
  };

  const parts = new Intl.DateTimeFormat("en-US", options).formatToParts(date);
  const getValue = (type: string) => parts.find((p) => p.type === type)?.value || "";

  return {
    year: getValue("year"),
    month: getValue("month"),
    day: getValue("day"),
    hour: getValue("hour"),
    minute: getValue("minute"),
    second: includeSeconds ? getValue("second") : "00",
  };
};

export default tool({
  description: "Get the current date or datetime in various formats (UTC or Denver local time)",
  args: {
    format: tool.schema
      .enum(["date", "datetime", "human"])
      .optional()
      .describe("Format: 'date' (YYYY-MM-DD), 'datetime' (ISO 8601), 'human' (human-readable). Default: 'date'"),
    localtime: tool.schema
      .boolean()
      .optional()
      .describe("Use Denver timezone (America/Denver) instead of UTC. Default: false"),
  },
  async execute(args) {
    const format: DateTimeFormat = args.format || "date";
    const useLocalTime = args.localtime ?? false;
    const timeZone = useLocalTime ? "America/Denver" : "UTC";
    const now = new Date();

    switch (format) {
      case "datetime": {
        if (useLocalTime) {
          const parts = formatParts(now, timeZone, true);
          const offset = buildOffset(now, timeZone);

          return `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}${offset}`;
        }

        return now.toISOString().replace(/\.\d{3}Z$/, "Z");
      }
      case "human": {
        const options: Intl.DateTimeFormatOptions = {
          weekday: "long",
          year: "numeric",
          month: "long",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          timeZone,
          timeZoneName: "short",
          hour12: false,
        };

        return now.toLocaleDateString("en-US", options);
      }
      case "date":
      default: {
        const parts = formatParts(now, timeZone);
        return `${parts.year}-${parts.month}-${parts.day}`;
      }
    }
  },
});
