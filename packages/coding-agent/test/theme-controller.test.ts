import type { TUI } from "@earendil-works/pi-tui";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SettingsManager } from "../src/core/settings-manager.ts";
import { initTheme, setRegisteredThemes, theme } from "../src/modes/interactive/theme/theme.ts";
import { InteractiveThemeController } from "../src/modes/interactive/theme/theme-controller.ts";

function createUi(): {
	ui: TUI;
	queryTerminalBackgroundColor: ReturnType<typeof vi.fn>;
} {
	const queryTerminalBackgroundColor = vi.fn();
	const ui = {
		invalidate: vi.fn(),
		requestRender: vi.fn(),
		setTerminalColorSchemeNotifications: vi.fn(),
		onTerminalColorSchemeChange: vi.fn(() => vi.fn()),
		queryTerminalBackgroundColor,
	} as unknown as TUI;
	return { ui, queryTerminalBackgroundColor };
}

function createSettingsManager(themeSetting: string | undefined): {
	settingsManager: SettingsManager;
	setTheme: ReturnType<typeof vi.fn>;
	flush: ReturnType<typeof vi.fn>;
} {
	const setTheme = vi.fn();
	const flush = vi.fn(async () => {});
	const settingsManager = {
		getThemeSetting: vi.fn(() => themeSetting),
		setTheme,
		flush,
	} as unknown as SettingsManager;
	return { settingsManager, setTheme, flush };
}

afterEach(() => {
	setRegisteredThemes([]);
	initTheme("dark");
});

describe("InteractiveThemeController", () => {
	it("applies the invocation override instead of settings", async () => {
		const { ui, queryTerminalBackgroundColor } = createUi();
		const { settingsManager, setTheme, flush } = createSettingsManager("light/dark");
		const controller = new InteractiveThemeController(ui, settingsManager, {
			showError: vi.fn(),
			onChanged: vi.fn(),
			themeOverride: "light",
		});

		expect(theme.name).toBe("light");
		await controller.applyFromSettings();

		expect(theme.name).toBe("light");
		expect(queryTerminalBackgroundColor).not.toHaveBeenCalled();
		expect(setTheme).not.toHaveBeenCalled();
		expect(flush).not.toHaveBeenCalled();
	});
});
