
# 3 "g_python_extraction.mlg"
 

open Pp
open Names
open Libnames
open Stdarg
open Miniml
open Common
open Python

(** [python_mode] is set to [true] by [Extraction Language Python.] *)
let python_mode = ref false

(** Build a monolithic .py file for the given global reference. *)
let python_extract ~opaque_access qid =
  let gr   = Smartlocate.global_with_alias qid in
  let ctx  = Environ.universes_of_global (Global.env ()) gr in
  let insts =
    let raw = InfvInst.generate ctx in
    if raw = [] then [InfvInst.empty] else raw
  in
  let globals = List.map (fun inst -> { glob = gr; inst }) insts in
  let state  = State.make ~modular:false ~library:false
                 ~keywords:Python.python_descr.keywords () in
  let struc  = Extract_env.mono_environment state ~opaque_access globals [] in
  let safe   = { mldummy  = false; tdummy   = false;
                 tunknown = false; magic    = false } in
  let base   = Id.to_string (Nametab.basename_of_global gr) in
  let name   = Id.of_string base in
  let pp     =
    Python.python_descr.preamble state name None DirPath.Set.empty safe ++
    Python.python_descr.pp_struct state struc
  in
  let fname  = base ^ ".py" in
  let oc     = open_out fname in
  output_string oc (Pp.string_of_ppcmds pp);
  close_out oc;
  Feedback.msg_notice (str "Extracted to " ++ str fname)


# 44 "g_python_extraction.ml"

let () = Vernacextend.static_vernac_extend ~plugin:(Some "rocq-python-extraction") ~command:"ExtractionLanguagePython" ~classifier:(fun ~atts:_ _ -> Vernacextend.classify_as_sideeff) ~ignore_kw:false ?entry:None 
         [(Vernacextend.TyML
         (false,
          Vernacextend.TyTerminal
          ("Extraction",
           Vernacextend.TyTerminal
           ("Language",
            Vernacextend.TyTerminal ("Python", Vernacextend.TyNil))),
          (let coqpp_body () = Vernactypes.vtdefault (fun () -> 
# 47 "g_python_extraction.mlg"
       python_mode := true 
# 57 "g_python_extraction.ml"
) in
            fun ?loc ~atts () ->
            coqpp_body (Attributes.unsupported_attributes atts)),
          None))]

let () = Vernacextend.static_vernac_extend ~plugin:(Some "rocq-python-extraction") ~command:"PythonExtraction" ~classifier:(fun ~atts:_ _ -> Vernacextend.classify_as_query) ~ignore_kw:false ?entry:None 
         [(Vernacextend.TyML
         (false,
          Vernacextend.TyTerminal
          ("Python",
           Vernacextend.TyTerminal
           ("Extraction",
            Vernacextend.TyNonTerminal (Extend.TUentry (Genarg.get_arg_tag wit_global),
            Vernacextend.TyNil))),
          (let coqpp_body x () =
            Vernactypes.vtopaqueaccess (fun ~opaque_access -> (
# 53 "g_python_extraction.mlg"
       python_extract x 
# 76 "g_python_extraction.ml"
)
            ~opaque_access) in fun x ?loc ~atts () ->
            coqpp_body x (Attributes.unsupported_attributes atts)),
          None))]

