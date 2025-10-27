{-# LANGUAGE OverloadedStrings #-}

module Main where

import System.Environment (getArgs)
import System.Exit (exitFailure)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO

import DSL.Parser
import DSL.AST

import Text.Megaparsec (errorBundlePretty)

main :: IO ()
main = do
    args <- getArgs
    filePath <- case args of
      (fp:_) -> return fp
      _ -> putStrLn "Usage: mydsl <file>" >> exitFailure

    -- Read the file
    content <- readFile filePath

    -- Parse tensor declarations
    case parseTensorDecls content of
      Left err -> do
        putStrLn "Parse error:"
        putStrLn (errorBundlePretty err)
      Right decls -> do
        putStrLn "Parsed tensor declarations:"
        mapM_ print decls
