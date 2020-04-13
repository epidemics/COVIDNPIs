// Creates CSV from data v3, you will have to download this yourself from
// https://storage.googleapis.com/static-covid/static/data-main-v3.json
// I uploaded this just for reference

import * as fs from "fs";
let {regions} = require("./data-main-v3.json")

let out = fs.openSync("./out.csv", "w");
fs.writeSync(out, "CodeISO3,TracesV3\n")
Object.keys(regions).forEach((key) => {
    let region = regions[key];
    fs.writeSync(out, `"${region.iso_alpha_3}","${region.data.infected_per_1000.traces_url}"\n`)
})
fs.closeSync(out);
