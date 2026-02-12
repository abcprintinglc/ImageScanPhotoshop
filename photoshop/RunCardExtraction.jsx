#target photoshop

(function () {
    var pythonPath = prompt("Python executable path:", "python");
    if (!pythonPath) {
        alert("Cancelled.");
        return;
    }

    var thisScript = new File($.fileName);
    var repoRoot = thisScript.parent.parent;
    var extractor = new File(repoRoot.fsName + "/business_card_extractor.py");
    if (!extractor.exists) {
        alert("Could not find business_card_extractor.py next to this repository.\nExpected: " + extractor.fsName);
        return;
    }

    var inputFile = File.openDialog("Select scan input (image / TIFF / PDF)");
    if (!inputFile) {
        alert("Cancelled.");
        return;
    }

    var outputFolder = Folder.selectDialog("Select output folder for card PNGs");
    if (!outputFolder) {
        alert("Cancelled.");
        return;
    }

    var cmd = '"' + pythonPath + '" "' + extractor.fsName + '" "' + inputFile.fsName + '" -o "' + outputFolder.fsName + '"';
    var result = app.system(cmd);
    alert("Business card extraction finished.\n\nOutput: " + outputFolder.fsName + "\n\nCommand output:\n" + result);
})();
