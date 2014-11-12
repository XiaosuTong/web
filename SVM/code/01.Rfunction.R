

rhchmod(path, permissions)



rhcp(ifile, ofile, delete = FALSE)



rhdel(folder)



rherrors(job, prefix = "rhipe_debug", num.file = 1)



rhex(conf, async = TRUE, mapred, ...)



rhexists(path)



rhfmt(type, ...)



rhget(src, dest)



rhinit()



rhIterator(files, type = "sequence", chunksize = 1000, chunk = "records",
  skip = rhoptions()$file.types.remove.regex, mc = lapply,
  textual = FALSE)
}



rhJobInfo(job)



rhkill(job)



rhls(folder = NULL, recurse = FALSE, nice = "h")



rhload(file, envir = parent.frame())



rhmap(expr = NULL, before = NULL, after = NULL)



rhmapfile(paths)



rhmkdir(path, permissions)



rhmv(ifile, ofile)



rhofolder(job)



rhoptions(li = NULL, ...)



rhput(src, dest, deletedest = TRUE)



rhread(files, type = c("sequence"), max = -1L,
  skip = rhoptions()$file.types.remove.regex, mc = lapply,
  textual = FALSE, verbose = TRUE, ...)



rhsave(..., file, envir = parent.frame())



rhsave.image(..., file)



rhstatus(job, mon.sec = 5, autokill = TRUE, showErrors = TRUE,
  verbose = FALSE, handler = rhoptions()$statusHandler)



rhuz(r)



rhwatch(map = NULL, reduce = NULL, combiner = FALSE, setup = NULL,
  cleanup = NULL, input = NULL, output = NULL, orderby = "bytes",
  mapred = NULL, shared = c(), jarfiles = c(), zips = c(),
  partitioner = NULL, copyFiles = FALSE, jobname = "",
  parameters = NULL, job = NULL, mon.sec = 5,
  readback = rhoptions()$readback, debug = NULL, noeval = FALSE, ...)
}



rhwrite(object, file, numfiles = 1, chunk = 1, passByte = 1024 * 1024 *
  20, kvpairs = TRUE, verbose = TRUE)


