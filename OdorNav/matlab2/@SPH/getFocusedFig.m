function out = getFocusedFig(out)
%GETFOCUSEDFIGURE Focus a figure if not deleted, regenerate window if it was

if exist('INP','var')
    if isgraphics(out)
        set(0, 'CurrentFigure', out);
        return
    end
end
out = figure;

end

