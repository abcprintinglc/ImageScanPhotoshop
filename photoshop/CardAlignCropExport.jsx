#target photoshop
app.bringToFront();

(function () {
    if (app.documents.length === 0) {
        alert("Open a document first.");
        return;
    }

    var doc = app.activeDocument;

    function hasSelection() {
        try {
            var b = doc.selection.bounds;
            return b && b.length === 4;
        } catch (e) {
            return false;
        }
    }

    if (!hasSelection()) {
        alert("Please make a selection around the card/image first, then run this script.");
        return;
    }

    var dlg = new Window("dialog", "Align, Crop (non-destructive), Export PNG");
    dlg.orientation = "column";
    dlg.alignChildren = "left";

    var angleGroup = dlg.add("group");
    angleGroup.add("statictext", undefined, "Rotate angle (degrees, clockwise negative):");
    var angleInput = angleGroup.add("edittext", undefined, "0");
    angleInput.characters = 8;

    var colorsGroup = dlg.add("group");
    colorsGroup.add("statictext", undefined, "PNG-8 colors (smaller file = fewer colors):");
    var colorsInput = colorsGroup.add("edittext", undefined, "64");
    colorsInput.characters = 6;

    var folderGroup = dlg.add("group");
    folderGroup.add("statictext", undefined, "Export folder:");
    var folderField = folderGroup.add("edittext", undefined, Folder.myDocuments.fsName);
    folderField.characters = 42;
    var browseBtn = folderGroup.add("button", undefined, "Browse");

    browseBtn.onClick = function () {
        var f = Folder.selectDialog("Choose output folder");
        if (f) {
            folderField.text = f.fsName;
        }
    };

    var keepOpen = dlg.add("checkbox", undefined, "Keep document open after export");
    keepOpen.value = true;

    var buttons = dlg.add("group");
    buttons.alignment = "right";
    buttons.add("button", undefined, "Cancel", { name: "cancel" });
    var okBtn = buttons.add("button", undefined, "Run", { name: "ok" });

    if (dlg.show() !== 1) {
        return;
    }

    var angle = parseFloat(angleInput.text);
    if (isNaN(angle)) angle = 0;

    var colors = parseInt(colorsInput.text, 10);
    if (isNaN(colors) || colors < 2) colors = 64;
    if (colors > 256) colors = 256;

    var outFolder = new Folder(folderField.text);
    if (!outFolder.exists) {
        outFolder.create();
    }

    function cropToSelectionKeepHiddenPixels() {
        var b = doc.selection.bounds;
        var left = b[0].as("px");
        var top = b[1].as("px");
        var right = b[2].as("px");
        var bottom = b[3].as("px");

        var idCrop = charIDToTypeID("Crop");
        var desc = new ActionDescriptor();
        var idT = charIDToTypeID("T   ");
        var rect = new ActionDescriptor();
        rect.putUnitDouble(charIDToTypeID("Top "), charIDToTypeID("#Pxl"), top);
        rect.putUnitDouble(charIDToTypeID("Left"), charIDToTypeID("#Pxl"), left);
        rect.putUnitDouble(charIDToTypeID("Btom"), charIDToTypeID("#Pxl"), bottom);
        rect.putUnitDouble(charIDToTypeID("Rght"), charIDToTypeID("#Pxl"), right);
        desc.putObject(idT, charIDToTypeID("Rctn"), rect);

        desc.putBoolean(stringIDToTypeID("delete"), false);
        executeAction(idCrop, desc, DialogModes.NO);
    }

    function exportSmallPng(fileObj, colorCount) {
        var opts = new ExportOptionsSaveForWeb();
        opts.format = SaveDocumentType.PNG;
        opts.PNG8 = true;
        opts.transparency = true;
        opts.interlaced = false;
        opts.optimized = true;
        opts.colors = colorCount;
        opts.includeProfile = false;

        doc.exportDocument(fileObj, ExportType.SAVEFORWEB, opts);
    }

    app.activeDocument.suspendHistory("Align/Crop/Export PNG", "runPipeline()");

    function runPipeline() {
        if (angle !== 0) {
            doc.rotateCanvas(angle);
        }

        cropToSelectionKeepHiddenPixels();

        var safeName = doc.name.replace(/\.[^\.]+$/, "").replace(/[\\\/:*?\"<>|]+/g, "_");
        var outFile = new File(outFolder.fsName + "/" + safeName + "_aligned.png");
        exportSmallPng(outFile, colors);

        if (!keepOpen.value) {
            doc.close(SaveOptions.DONOTSAVECHANGES);
        }

        alert("Exported: " + outFile.fsName);
    }
})();
