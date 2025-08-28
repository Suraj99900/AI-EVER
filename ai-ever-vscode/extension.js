const vscode = require("vscode");
const { fetch } = require("undici");

// 🔹 Helper: call backend
async function callAIBackend(payload) {
    try {
        const res = await fetch("http://127.0.0.1:5000/complete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        return await res.json();
    } catch (err) {
        vscode.window.showErrorMessage("AI-EVER backend error: " + err.message);
        return null;
    }
}

// 🔹 Command: Manual code completion
async function completeWithAI() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const document = editor.document;
    const selection = editor.selection;
    const context = selection.isEmpty
        ? document.getText()
        : document.getText(selection);

    const data = await callAIBackend({
        context,
        task: "completion",
        stream: false,
    });

    if (data && data.status === "success") {
        editor.edit(editBuilder => {
            if (selection.isEmpty) {
                editBuilder.insert(selection.start, data.completion);
            } else {
                editBuilder.replace(selection, data.completion);
            }
        });
    } else if (data) {
        vscode.window.showErrorMessage("AI-EVER Error: " + data.message);
    }
}

// 🔹 Command: Fix selected code
async function fixCodeWithAI() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const selection = editor.selection;
    if (selection.isEmpty) {
        vscode.window.showErrorMessage("Select some buggy code to fix.");
        return;
    }

    const buggyCode = editor.document.getText(selection);
    const data = await callAIBackend({
        context: buggyCode,
        task: "bug_fix",
        stream: false,
    });

    if (data && data.status === "success") {
        editor.edit(editBuilder => {
            editBuilder.replace(selection, data.completion);
        });
    } else if (data) {
        vscode.window.showErrorMessage("AI-EVER Error: " + data.message);
    }
}

// 🔹 Inline ghost completions (like Tabnine)
const inlineProvider = {
    async provideInlineCompletionItems(document, position) {
        const codeBefore = document.getText(
            new vscode.Range(new vscode.Position(0, 0), position)
        );

        const data = await callAIBackend({
            context: codeBefore,
            task: "completion",
            stream: false,
        });

        if (data && data.status === "success") {
            return [
                {
                    insertText: data.completion,
                    range: new vscode.Range(position, position),
                },
            ];
        }
        return [];
    },
};

// 🔹 Activate extension
function activate(context) {
    console.log("AI-EVER extension activated ✅");

    // Command: complete code
    let disposableComplete = vscode.commands.registerCommand(
        "ai-ever-vscode.completeCode",
        completeWithAI
    );

    // Command: fix code
    let disposableFix = vscode.commands.registerCommand(
        "ai-ever-vscode.fixCode",
        fixCodeWithAI
    );

    // Inline completions
    let inlineProviderDisposable = vscode.languages.registerInlineCompletionItemProvider(
        { pattern: "**" },
        inlineProvider
    );

    context.subscriptions.push(disposableComplete);
    context.subscriptions.push(disposableFix);
    context.subscriptions.push(inlineProviderDisposable);
}

// 🔹 Deactivate extension
function deactivate() {}

module.exports = {
    activate,
    deactivate,
};
